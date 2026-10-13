"""Сбор таблицы с учебного сайта. Запуск: python collect_data.py."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup


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


if __name__ == "__main__":
    collect_hockey()
