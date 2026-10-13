"""Полный код решения задания NHL в одном Python-файле.

Запуск: python hockey_solution.py
Библиотеки: python -m pip install -r requirements.txt

При наличии data/hockey_raw.csv используется сохранённая таблица.
Для повторного сбора поменяйте COLLECT_AGAIN на True.
Таблицы выводятся в терминал, графики и результаты сохраняются в results/.
Ноутбук hockey_solution.ipynb этот скрипт не читает и не изменяет.
"""

import os
import matplotlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import hashlib
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup

# Пути к данным считаем от папки с этим скриптом.
os.chdir(Path(__file__).resolve().parent)


def show_table(table):
    # Вместо отображения в Jupyter печатаем таблицу целиком в терминал.
    print(table.to_string())
    print()

# Импорты и настройки

pd.set_option("display.max_columns", 20)
pd.set_option("display.precision", 4)
plt.rcParams["figure.figsize"] = (8, 4)
Path("results").mkdir(exist_ok=True)

# Сбор данных и чтение CSV




def collect_hockey():
    url = "https://www.scrapethissite.com/pages/forms/"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    response = session.get(url, timeout=60)
    response.raise_for_status()
    first_page = BeautifulSoup(response.text, "html.parser")

    # Номера страниц берём из ссылок, а не задаём вручную.
    page_numbers = []
    for link in first_page.select(".pagination a"):
        query = parse_qs(urlparse(link["href"]).query)
        if "page_num" in query:
            page_numbers.append(int(query["page_num"][0]))
    if not page_numbers:
        raise ValueError("Не найдены ссылки на страницы")
    last_page = max(page_numbers)

    columns = ["team", "year", "wins", "losses", "ot_losses", "win_rate",
               "goals_for", "goals_against", "goal_diff"]
    rows = []
    pages = []

    for page in range(1, last_page + 1):
        if page == 1:
            soup = first_page
        else:
            time.sleep(0.3)
            response = session.get(url, params={"page_num": page}, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

        page_rows = []
        for tr in soup.select("tr.team"):
            values = [td.get_text(strip=True) for td in tr.select("td")]
            if len(values) != len(columns):
                raise ValueError(f"На странице {page} изменилась структура таблицы")
            page_rows.append(values)
        if not page_rows:
            raise ValueError(f"Страница {page} оказалась пустой")

        rows.extend(page_rows)
        pages.append({"page": page, "rows": len(page_rows),
                      "first_team": page_rows[0][0], "first_year": int(page_rows[0][1]),
                      "last_team": page_rows[-1][0], "last_year": int(page_rows[-1][1])})
        print(f"Страница {page}/{last_page}: {len(page_rows)} строк", flush=True)

    data = pd.DataFrame(rows, columns=columns)
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column].replace("", None), errors="raise")

    if data.duplicated(["team", "year"]).any():
        raise ValueError("Найдены повторяющиеся пары команда-год; выгрузка не сохранена")

    Path("data").mkdir(exist_ok=True)
    csv_path = Path("data/hockey_raw.csv")
    data.to_csv(csv_path, index=False)
    metadata = {
        "url": url,
        "collected_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "pages_processed": last_page,
        "rows": len(data),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "pages": pages,
    }
    Path("data/collection_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data


# При повторном запуске используем сохранённую выгрузку.
# Для нового сбора нужно поменять False на True.
COLLECT_AGAIN = False
if COLLECT_AGAIN or not Path("data/hockey_raw.csv").exists():
    collect_hockey()

raw = pd.read_csv("data/hockey_raw.csv")
info = json.loads(Path("data/collection_info.json").read_text())
page_log = pd.DataFrame(info["pages"])
assert len(raw) == info["rows"] == page_log["rows"].sum()
assert hashlib.sha256(Path("data/hockey_raw.csv").read_bytes()).hexdigest() == info["sha256"]
print("Источник:", info["url"])
print("Дата сбора:", info["collected_at"])
print("Страниц:", info["pages_processed"], "Строк:", len(raw))
print("Годы:", raw["year"].min(), "—", raw["year"].max())
show_table(page_log)
show_table(raw.head())

# Проверка качества таблицы

quality = pd.DataFrame({"type": raw.dtypes.astype(str), "missing": raw.isna().sum()})
show_table(quality)

count_columns = ["wins", "losses", "ot_losses", "goals_for", "goals_against"]
checks = pd.Series({
    "Дубликаты команда-год": int(raw.duplicated(["team", "year"]).sum()),
    "Неверная разность голов": int((raw["goal_diff"] != raw["goals_for"] - raw["goals_against"]).sum()),
    "Доля побед вне [0, 1]": int((~raw["win_rate"].between(0, 1)).sum()),
    "Отрицательные счётчики": int((raw[count_columns] < 0).sum().sum()),
    "Нецелые счётчики": int((raw[count_columns].drop(columns="ot_losses") % 1 != 0).sum().sum()),
    "Нецелые известные OT-поражения": int((raw["ot_losses"].dropna() % 1 != 0).sum()),
})
show_table(checks)
assert (checks == 0).all()
assert raw.drop(columns="ot_losses").notna().all().all()
assert raw["team"].str.strip().ne("").all()

by_year = raw.groupby("year").agg(
    teams=("team", "size"),
    missing_ot=("ot_losses", lambda values: values.isna().sum()),
    median_win_rate=("win_rate", "median"),
)
show_table(by_year)
missing_years = sorted(set(range(raw["year"].min(), raw["year"].max() + 1)) - set(raw["year"]))
print("Пропущенные годы:", missing_years)

# Построение target и истории команды

# Медиана считается по всем командам года, ещё до отбора пар.
medians = raw.groupby("year")["win_rate"].median()
future = raw[["team", "year", "win_rate"]].rename(columns={
    "year": "target_year", "win_rate": "future_win_rate"
})
future["future_median"] = future["target_year"].map(medians)

current = raw.copy()
current["target_year"] = current["year"] + 1
joined = current.merge(future, on=["team", "target_year"], how="left", validate="one_to_one")
excluded = joined[joined["future_win_rate"].isna()].copy()
excluded["reason"] = "Нет той же команды в следующем году"
excluded.loc[~excluded["target_year"].isin(raw["year"]), "reason"] = "Следующего года нет в таблице"
print("Исключено строк:", len(excluded))
show_table(excluded.groupby(["year", "reason"]).size().rename("rows").reset_index())

pairs = joined.dropna(subset=["future_win_rate"]).copy()
pairs["target"] = (pairs["future_win_rate"] > pairs["future_median"]).astype(int)
assert (pairs["target_year"] == pairs["year"] + 1).all()
assert not pairs.duplicated(["team", "year"]).any()
assert np.allclose(pairs["future_median"], pairs["target_year"].map(medians))
ties = pairs[pairs["future_win_rate"] == pairs["future_median"]]
assert (ties["target"] == 0).all()
print("Пар:", len(pairs), "Равенств медиане:", len(ties))

# Три ручных примера: выше, ровно и ниже медианы.
examples = pd.concat([
    pairs[pairs["target"] == 1].head(1),
    ties.head(1),
    pairs[pairs["future_win_rate"] < pairs["future_median"]].head(1),
])
show_table(examples[["team", "year", "target_year", "win_rate", "future_win_rate", "future_median", "target"]])
for _, row in examples.iterrows():
    values = sorted(raw.loc[raw["year"] == row["target_year"], "win_rate"].tolist())
    middle = len(values) // 2
    if len(values) % 2 == 1:
        manual_median = values[middle]
    else:
        manual_median = (values[middle - 1] + values[middle]) / 2
    manual_target = int(row["future_win_rate"] > manual_median)
    assert manual_target == row["target"]
    print(f'{row["team"]}, {row["year"]} → {row["target_year"]}: '
          f'{row["future_win_rate"]} > {manual_median}? target = {manual_target}; '
          f'медиана из всех {len(values)} команд года')

# История команды: только ровно предыдущий год, без заполнения через разрывы.
previous = raw[["team", "year", "win_rate"]].rename(columns={"win_rate": "previous_win_rate"})
previous["year"] = previous["year"] + 1
pairs = pairs.merge(previous, on=["team", "year"], how="left", validate="one_to_one")
excluded.to_csv("results/excluded_rows.csv", index=False)

# Разделение по годам

train = pairs[pairs["target_year"] <= 2005].copy()
validation = pairs[pairs["target_year"].between(2006, 2008)].copy()
test = pairs[pairs["target_year"].between(2009, 2011)].copy()

parts = {"train": train, "validation": validation, "test": test}
split_rows = []
for name, part in parts.items():
    split_rows.append({"part": name, "target_years": sorted(part["target_year"].unique().tolist()),
                       "rows": len(part), "class_1_share": part["target"].mean()})
    assert part["target"].nunique() == 2
split_table = pd.DataFrame(split_rows)
show_table(split_table)
assert len(train) + len(validation) + len(test) == len(pairs)
assert train["target_year"].max() < validation["target_year"].min()
assert validation["target_year"].max() < test["target_year"].min()

# Вопросы к данным

# Вопрос 1: связана ли текущая доля побед с успехом в следующем году?
# Ожидаем положительную связь. Это поможет выбрать основной признак.
# Вопрос 2: полезен ли прошлый сезон при похожем текущем результате?
# Ожидаем, что хорошая история повышает долю будущего успеха.
# Для анализа используем только train.
print("EDA: только train, целевые сезоны до 2005 включительно")

# Исследование данных

eda = train.copy()
eda["current_group"] = pd.qcut(eda["win_rate"], q=4, duplicates="drop")
current_table = eda.groupby("current_group", observed=True).agg(
    rows=("target", "size"), success_share=("target", "mean"),
    mean_current=("win_rate", "mean"), mean_future=("future_win_rate", "mean"),
)
show_table(current_table)
correlation = eda["win_rate"].corr(eda["future_win_rate"])
print("Корреляция текущего и будущего win_rate:", round(correlation, 4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(len(current_table)), current_table["success_share"])
axes[0].set_xticks(range(len(current_table)), ["Низкая", "Ниже средней", "Выше средней", "Высокая"])
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Группа текущей доли побед")
axes[0].set_ylabel("Доля target = 1")
axes[0].set_title("Рис. 1а. Текущий успех и следующий сезон (train)")
axes[1].scatter(eda["win_rate"], eda["future_win_rate"], alpha=0.4, s=20)
axes[1].set_xlabel("Доля побед в сезоне t")
axes[1].set_ylabel("Доля побед в сезоне t+1")
axes[1].set_title("Рис. 1б. Связь двух сезонов (train)")
fig.tight_layout()
fig.savefig("results/current_win_rate.png", dpi=150)
plt.close()

history = train.dropna(subset=["previous_win_rate"]).copy()
history["current_group"] = pd.qcut(history["win_rate"], q=3, duplicates="drop")
history["previous_median"] = (history["year"] - 1).map(medians)
history["previous_group"] = np.where(history["previous_win_rate"] >= history["previous_median"],
                                     "Не ниже медианы", "Ниже медианы")
history_table = history.groupby(["current_group", "previous_group"], observed=True).agg(
    rows=("target", "size"), success_share=("target", "mean")
)
show_table(history_table)
print("Строк train без предыдущего сезона:", train["previous_win_rate"].isna().sum())

history_table["success_share"].unstack().plot(kind="bar", ylim=(0, 1), rot=0, figsize=(10, 4))
plt.title("Рис. 2. Предыдущий сезон при похожем текущем результате (train)")
plt.xlabel("Группа текущей доли побед")
plt.ylabel("Доля target = 1")
plt.tight_layout()
plt.savefig("results/previous_win_rate.png", dpi=150)
plt.close()

team_example = train[train["team"] == "Boston Bruins"]
show_table(team_example[["team", "year", "previous_win_rate", "win_rate", "future_win_rate", "future_median", "target"]].head(6))

# Простые ориентиры

dummy = DummyClassifier(strategy="prior")
dummy.fit(train[["win_rate"]], train["target"])
constant_validation = dummy.predict_proba(validation[["win_rate"]])[:, 1]
current_validation = validation["win_rate"].to_numpy()

baseline_table = pd.DataFrame({
    "model": ["Константа", "Текущий win_rate"],
    "validation_roc_auc": [
        roc_auc_score(validation["target"], constant_validation),
        roc_auc_score(validation["target"], current_validation),
    ],
})
show_table(baseline_table)
print("Постоянная вероятность по train:", round(train["target"].mean(), 4))

# Первая модель

features = ["win_rate", "goal_diff"]

def make_model():
    return Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
    ])

model = make_model()
model.fit(train[features], train["target"])
validation_probability = model.predict_proba(validation[features])[:, 1]
first_auc = roc_auc_score(validation["target"], validation_probability)
print("Признаки:", features)
print("ROC-AUC первой модели на validation:", round(first_auc, 4))
show_table(pd.concat([baseline_table, pd.DataFrame({
    "model": ["Логистическая регрессия"], "validation_roc_auc": [first_auc]
})], ignore_index=True))

# Эксперимент и выбор модели

# Ожидаем небольшое улучшение: два сезона могут лучше показать силу команды.
# Меняем только состав признаков: добавляем результат ровно предыдущего года.
experiment_features = features + ["previous_win_rate"]
experiment_model = make_model()
experiment_model.fit(train[experiment_features], train["target"])
experiment_probability = experiment_model.predict_proba(validation[experiment_features])[:, 1]
experiment_auc = roc_auc_score(validation["target"], experiment_probability)

experiments = pd.DataFrame({
    "model": ["Логистическая регрессия", "Регрессия с предыдущим сезоном"],
    "validation_roc_auc": [first_auc, experiment_auc],
})
show_table(experiments)

# Правило выбора: больше ROC-AUC на validation. При равенстве — простая модель.
if experiment_auc > first_auc:
    final_model = experiment_model
    final_features = experiment_features
    final_name = "Регрессия с предыдущим сезоном"
    final_validation_auc = experiment_auc
else:
    final_model = model
    final_features = features
    final_name = "Логистическая регрессия"
    final_validation_auc = first_auc

choice = {
    "model": final_name, "features": final_features,
    "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42,
    "imputer": "median (train only)", "scaler": "StandardScaler (train only)",
    "threshold": 0.5, "validation_roc_auc": final_validation_auc,
    "selection_rule": "Наибольший validation ROC-AUC; при равенстве исходная модель",
    "fit_data": "train only; no refit after selection",
}
Path("results/model_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=2))
print("Выбор зафиксирован до оценки test:")
print(json.dumps(choice, ensure_ascii=False, indent=2))
experiments.to_csv("results/validation_models.csv", index=False)

# Итоговая оценка на test

# Первое использование test для оценки выбранного решения.
# Все модели уже обучены, признаки и порог уже зафиксированы.
test_probability = final_model.predict_proba(test[final_features])[:, 1]
constant_test = dummy.predict_proba(test[["win_rate"]])[:, 1]
current_test = test["win_rate"].to_numpy()

def metrics_row(name, probability):
    prediction = (probability >= 0.5).astype(int)
    return {"model": name, "ROC_AUC": roc_auc_score(test["target"], probability),
            "F1": f1_score(test["target"], prediction, zero_division=0),
            "accuracy": accuracy_score(test["target"], prediction)}

test_metrics = pd.DataFrame([
    metrics_row("Константа", constant_test),
    metrics_row("Текущий win_rate", current_test),
    metrics_row(final_name, test_probability),
])
show_table(test_metrics)
print("ROC-AUC выбранной модели: validation =", round(final_validation_auc, 4),
      "test =", round(test_metrics.iloc[2]["ROC_AUC"], 4))
print("Матрица ошибок: строки — факт 0/1, столбцы — прогноз 0/1")
show_table(pd.DataFrame(confusion_matrix(test["target"], test_probability >= 0.5, labels=[0, 1]),
                     index=["Факт 0", "Факт 1"], columns=["Прогноз 0", "Прогноз 1"]))

test_results = test.copy()
test_results["probability"] = test_probability
test_results["prediction"] = (test_probability >= 0.5).astype(int)
test_results["correct"] = test_results["prediction"] == test_results["target"]
test_results.to_csv("results/test_predictions.csv", index=False)
test_metrics.to_csv("results/test_metrics.csv", index=False)

# Разбор ошибок

# Принцип отбора: пять самых уверенных неверных ответов.
errors = test_results[~test_results["correct"]].copy()
errors["confidence"] = np.where(errors["prediction"] == 1,
                                errors["probability"], 1 - errors["probability"])
errors["error_type"] = np.where(errors["prediction"] == 1, "Ложный успех", "Пропущенный успех")
errors["future_gap"] = errors["future_win_rate"] - errors["future_median"]
top_errors = errors.sort_values(["confidence", "team", "year"], ascending=[False, True, True]).head(5)
error_columns = ["team", "year", "target_year", "win_rate", "goal_diff", "previous_win_rate",
                 "probability", "prediction", "target", "future_win_rate", "future_median",
                 "future_gap", "confidence", "error_type"]
show_table(top_errors[error_columns])
print("Всего ошибок:", len(errors), "из", len(test_results))
show_table(errors.groupby("error_type").size().rename("rows"))
print("Среди пяти разобранных ошибок:")
show_table(top_errors.groupby("error_type").size().rename("rows"))
top_errors[error_columns].to_csv("results/top_errors.csv", index=False)

# Сохранение итогов

# Сохраняем короткий итог и числа для воспроизводимости.
summary = {
    "rows": len(raw), "pages": info["pages_processed"], "missing_ot": int(raw["ot_losses"].isna().sum()),
    "missing_years": missing_years, "pairs": len(pairs), "excluded": len(excluded), "ties": len(ties),
    "split": split_table.to_dict(orient="records"), "correlation": correlation,
    "current_groups": current_table.reset_index().assign(current_group=lambda x: x["current_group"].astype(str)).to_dict(orient="records"),
    "history_groups": history_table.reset_index().assign(current_group=lambda x: x["current_group"].astype(str)).to_dict(orient="records"),
    "baselines": baseline_table.to_dict(orient="records"), "first_auc": first_auc,
    "experiment_auc": experiment_auc, "choice": choice, "test_metrics": test_metrics.to_dict(orient="records"),
    "error_count": len(errors), "top_errors": json.loads(top_errors[error_columns].to_json(orient="records")),
    "team_example": json.loads(team_example.head(6).to_json(orient="records")),
}
Path("results/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("Финальная модель:", final_name)
print("Признаки:", final_features)
print("Порог: 0.5. Настройки и результаты сохранены в results/.")
