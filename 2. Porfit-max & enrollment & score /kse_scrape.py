#!/usr/bin/env python3
"""
abit-poisk.org.ua — KSE Bachelor Applicants Scraper
====================================================
Крок 1: Збирає всі конкурсні пропозиції (Б) з КШЕ
Крок 2: По кожній пропозиції збирає список абітурієнтів (з пагінацією)
Крок 3: По кожному абітурієнту заходить в пошук та витягує усі дані

Залежності:
    pip install playwright beautifulsoup4 requests lxml
    playwright install chromium

Запуск:
    python kse_scraper.py

Результат:
    kse_students.csv  — список студентів по напрямках (крок 2)
    kse_profiles.csv  — профілі студентів, дописується рядок за рядком (крок 3)
    kse_progress.json — чекпоінт кроків 1–2 (для швидкого відновлення)
"""

import asyncio
import argparse
import csv
import json
import re
import random
import time
import sys
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Конфіг ───────────────────────────────────────────────────────────────────

BASE_URL    = "https://abit-poisk.org.ua"
UNIVER_URL  = f"{BASE_URL}/rate2025/univer/3924"

# CSV_STUDENTS — всі студенти по напрямках (крок 2, пишеться по завершенні кроку 2)
# CSV_PROFILES — профілі, пишеться по одному рядку одразу після парсингу (крок 3)
CSV_STUDENTS  = Path("kse_students.csv")
CSV_PROFILES  = Path("kse_profiles.csv")
PROGRESS_FILE = Path("kse_progress.json")
API_CACHE     = Path("kse_api.json")   # знайдений endpoint після discovery
LOG_FILE      = Path("kse_scraper.log")

# Фіксовані колонки для CSV профілів — 1 рядок = 1 заявка
PROFILE_CSV_COLS = [
    # Ідентифікація студента
    "student_name",
    # КШЕ-контекст (з рейтингового списку, крок 2)
    "kse_direction", "kse_subspecialty", "kse_program",
    "kse_list_position", "kse_priority", "kse_list_status",
    "kse_total_score", "kse_comp_scores", "kse_quota",
    "kse_direction_url",
    # Дані конкретної заявки (крок 3, з пошуку)
    "app_index",          # порядковий номер заявки для цього студента (1..N)
    "app_total_apps",     # всього заявок у студента
    "app_okr",            # ОКР: Б/М
    "app_status",         # С: Відраховано / Відхилено (бюджет) / тощо
    "app_position",       # №: позиція в рейтинговому списку
    "app_priority",       # П: пріоритет заявки
    "app_vm",             # Місця: загальна кількість місць (ВМ)
    "app_bm_max",         # Місця: бюджетних місць (БМmax)
    "app_k",              # Місця: кількість (К)
    "app_total_score",    # Σ: загальний конкурсний бал
    "app_sbo",            # СБО: середній бал освітнього документа
    # Складові конкурсного балу — числа і предмети окремо
    "app_comp_scores",         # числа через | (182|200|187)
    "app_comp_named",          # "Українська мова:182|Математика:200|Іноземна мова:187"
    "app_university",     # ВНЗ
    "app_faculty",        # Ф
    "app_specialty_code", # код спеціальності (121, 113, 033...)
    "app_specialty",      # назва спеціальності
    "app_quota",          # Кв
    "app_d",              # Д: остання колонка (+/-)
    "is_kse",             # YES якщо scores збіглись
    "search_url",
]

STUDENTS_CSV_COLS = [
    "direction_url", "direction_title", "subspecialty",
    # метадані напрямку з заголовку сторінки
    "dir_vm", "dir_bm_max", "dir_k", "dir_zayav", "dir_konkurs",
    "dir_kvota1", "dir_kvota2", "dir_score_components", "dir_form", "dir_duration",
    # дані студента
    "name", "search_url", "position", "priority", "status",
    "total_score", "scores", "quota",
]

# Rate limit Playwright (форма): 15с мінімум за правилами сайту
SEARCH_DELAY_MIN = 15
SEARCH_DELAY_MAX = 20
# Rate limit для прямих API запитів — набагато менше
API_DELAY_MIN  = 1.0
API_DELAY_MAX  = 2.5
PAGE_DELAY_MIN = 2
PAGE_DELAY_MAX = 5

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# ─── Утіліти ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text("utf-8"))
    return {
        "directions": [],  # [{url, title, subspecialty}]
        "students":   {},  # dir_url -> {direction, list: [...]}
        "done_dirs":  [],  # dir_url's вже зроблені
    }


def save_progress(data: dict):
    PROGRESS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
    )


# --- CSV utilities ------------------------------------------------------------

def csv_get_done_profiles() -> set:
    """Повертає search_url вже записані в CSV (для resume кроку 3)."""
    if not CSV_PROFILES.exists():
        return set()
    done = set()
    with open(CSV_PROFILES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("search_url", "").strip()
            if url:
                done.add(url)
    log(f"  CSV resume: {len(done)} профілів вже є в {CSV_PROFILES}")
    return done


def csv_ensure_profiles_header():
    """Створює заголовок CSV профілів якщо файл ще не існує."""
    if not CSV_PROFILES.exists():
        with open(CSV_PROFILES, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PROFILE_CSV_COLS).writeheader()


def csv_append_profile(student: dict, direction: dict, profile):
    """
    Дописує рядки у kse_profiles.csv — по одному рядку на кожну заявку студента.
    Якщо profile порожній або помилка — один рядок з ERROR.
    """
    base = {col: "" for col in PROFILE_CSV_COLS}
    base["student_name"]       = student.get("name", "")
    base["kse_direction"]      = direction.get("title", "")
    base["kse_subspecialty"]   = direction.get("subspecialty", "")
    base["kse_program"]        = direction.get("meta", {}).get("program", "")
    base["kse_list_position"]  = student.get("position", "")
    base["kse_priority"]       = student.get("priority", "")
    base["kse_list_status"]    = student.get("status", "")
    base["kse_total_score"]    = student.get("total_score", "")
    base["kse_comp_scores"]    = "|".join(str(s) for s in student.get("scores", []))
    base["kse_quota"]          = student.get("quota", "")
    base["kse_direction_url"]  = student.get("_dir_url", direction.get("url", ""))
    base["search_url"]         = student.get("search_url", "")

    with open(CSV_PROFILES, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_CSV_COLS)

        if profile is None or (isinstance(profile, dict) and profile.get("error")):
            row = {**base, "app_index": "ERROR", "app_total_apps": "0", "is_kse": ""}
            writer.writerow(row)
            return

        all_rows  = profile.get("all_rows", [])
        kse_scores_set = set(student.get("scores", []))
        total_apps = len(all_rows)

        if not all_rows:
            row = {**base, "app_index": "NO_DATA", "app_total_apps": "0", "is_kse": ""}
            writer.writerow(row)
            return

        for i, app in enumerate(all_rows, 1):
            row = dict(base)
            row["app_index"]      = str(i)
            row["app_total_apps"] = str(total_apps)

            hm    = app.get("headers_map", {})
            cells = app.get("cells", [])
            raw   = app.get("raw_text", "")

            # ── Хелпер: шукає в headers_map за ключовим словом ──────────────
            def h(*keys):
                for k in keys:
                    for hk, hv in hm.items():
                        if k.lower() in str(hk).lower() and hv:
                            return str(hv)
                return ""

            # ── Позиційний парсинг якщо headers відсутні ─────────────────────
            # Структура таблиці (за скріншотом):
            # 0=ОКР  1=ПІБ  2=С  3=№  4=П  5=Місця  6=Σ  7=СБО  8=Скл.бал  9=ВНЗ  10=Ф  11=Спец  12=Кв  13=Д
            def cell(idx: int) -> str:
                return cells[idx].strip() if idx < len(cells) else ""

            # ── ОКР ──────────────────────────────────────────────────────────
            row["app_okr"] = h("окр", "okr") or cell(0)

            # ── Статус ───────────────────────────────────────────────────────
            row["app_status"] = h("статус", "С") or cell(2)

            # ── Позиція №  ───────────────────────────────────────────────────
            row["app_position"] = h("місце", "№", "позиц") or cell(3)

            # ── Пріоритет П ──────────────────────────────────────────────────
            row["app_priority"] = h("пріор", "П", "prior") or cell(4)

            # ── Місця: ВМ / БМmax / К ────────────────────────────────────────
            places_raw = app.get("places_raw", "") or h("місця", "вм", "БМ") or cell(5)
            m = re.search(r'ВМ\s+(\d+)', places_raw, re.I)
            row["app_vm"] = m.group(1) if m else ""
            m = re.search(r'БМ[Mm]ax\s+(\d+)', places_raw, re.I)
            row["app_bm_max"] = m.group(1) if m else ""
            m = re.search(r'К\s+(\d+)', places_raw)
            row["app_k"] = m.group(1) if m else ""

            # ── Загальний бал Σ ───────────────────────────────────────────────
            row["app_total_score"] = app.get("total_score", "") or ""
            if not row["app_total_score"]:
                score_cell = cell(6)
                tm = re.search(r'\b(\d{3}\.\d{3})\b', score_cell)
                if tm:
                    row["app_total_score"] = tm.group(1)

            # ── СБО ──────────────────────────────────────────────────────────
            row["app_sbo"] = app.get("sbo", "") or (h("СБО", "сбо") or cell(7)).replace("—","").replace("-","").strip()

            # ── Складові конкурсного балу: числа і предмети ───────────────────
            comp_scores = app.get("comp_scores", [])
            comp_named  = app.get("comp_named", {})

            row["app_comp_scores"] = "|".join(str(s) for s in comp_scores)
            if comp_named:
                row["app_comp_named"] = "|".join(f"{k}:{v}" for k, v in comp_named.items())
            else:
                row["app_comp_named"] = row["app_comp_scores"]

            # ── ВНЗ ──────────────────────────────────────────────────────────
            row["app_university"] = (
                app.get("university", "") or
                h("внз", "університет", "заклад", "вуз") or cell(9)
            )

            # ── Факультет Ф ──────────────────────────────────────────────────
            row["app_faculty"] = h("факультет", "Ф", "фак") or cell(10)

            # ── Спеціальність: код + назва ────────────────────────────────────
            if app.get("spec_code"):
                row["app_specialty_code"] = app["spec_code"]
                row["app_specialty"]      = app.get("spec_name", "")
            else:
                spec_raw = h("спец", "напрям", "спеціаль") or cell(11)
                sc = re.match(r'^(\d{3})\s*(.*)', spec_raw)
                if sc:
                    row["app_specialty_code"] = sc.group(1)
                    row["app_specialty"]      = sc.group(2).strip()
                else:
                    row["app_specialty_code"] = ""
                    row["app_specialty"]      = spec_raw

            # ── Квота Кв ─────────────────────────────────────────────────────
            row["app_quota"] = h("квота", "кв") or cell(12)

            # ── Д ────────────────────────────────────────────────────────────
            row["app_d"] = app.get("app_d", "") or h("Д") or cell(13)

            # ── КШЕ match ────────────────────────────────────────────────────
            app_scores_set = set(comp_scores)
            if kse_scores_set and app_scores_set and kse_scores_set <= app_scores_set:
                row["is_kse"] = "YES"
            else:
                row["is_kse"] = "NO"

            writer.writerow(row)


def csv_write_students(progress: dict):
    """Записує/оновлює kse_students.csv з усіма студентами (крок 2)."""
    with open(CSV_STUDENTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STUDENTS_CSV_COLS)
        writer.writeheader()
        for dir_url, dir_data in progress["students"].items():
            direction = dir_data["direction"]
            meta = dir_data.get("meta", {})
            for s in dir_data.get("list", []):
                writer.writerow({
                    "direction_url":        dir_url,
                    "direction_title":      direction.get("title", ""),
                    "subspecialty":         direction.get("subspecialty", ""),
                    "dir_vm":               meta.get("vm", ""),
                    "dir_bm_max":           meta.get("bm_max", ""),
                    "dir_k":                meta.get("k", ""),
                    "dir_zayav":            meta.get("zayav", ""),
                    "dir_konkurs":          meta.get("konkurs", ""),
                    "dir_kvota1":           meta.get("kvota1", ""),
                    "dir_kvota2":           meta.get("kvota2", ""),
                    "dir_score_components": meta.get("score_components", ""),
                    "dir_form":             meta.get("form", ""),
                    "dir_duration":         meta.get("duration", ""),
                    "name":                 s.get("name", ""),
                    "search_url":           s.get("search_url", ""),
                    "position":             s.get("position", ""),
                    "priority":             s.get("priority", ""),
                    "status":               s.get("status", ""),
                    "total_score":          s.get("total_score", ""),
                    "scores":               "|".join(str(x) for x in s.get("scores", [])),
                    "quota":                s.get("quota", ""),
                })
    log(f"  Saved {CSV_STUDENTS}")


def is_rate_limited(text: str) -> bool:
    return "обмежена" in text or "search_off" in text


# ─── Крок 1: Список конкурсних пропозицій ────────────────────────────────────

def fetch_directions() -> list[dict]:
    """
    Парсить сторінку університету та витягує всі Б (бакалаврат) напрямки.
    Повертає: [{url, title, subspecialty, okr}]
    """
    log(f"GET {UNIVER_URL}")
    r = requests.get(
        UNIVER_URL,
        headers={"User-Agent": UA, "Accept-Language": "uk-UA,uk;q=0.9"},
        timeout=20,
        verify=False,
    )
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    directions = []
    seen = set()

    # Шукаємо всі рядки таблиці або блоки з посиланнями на /rate2025/direction/
    for link in soup.select("a[href*='/rate2025/direction/']"):
        href = link["href"]
        full_url = (BASE_URL + href) if href.startswith("/") else href
        if full_url in seen:
            continue

        # Беремо батьківський рядок і перевіряємо ОКР
        parent = link.find_parent("tr") or link.find_parent("div") or link.find_parent("li")
        parent_text = parent.get_text() if parent else ""

        # Фільтр: тільки Бакалаврат (ПЗСО)
        is_bachelor = (
            "Бакалавр" in parent_text and
            "ПЗСО" in parent_text and
            "Магістр" not in parent_text and
            "Доктор" not in parent_text
        )
        if not is_bachelor:
            continue

        okr = "Б"
        title = link.get_text(strip=True)

        # Підспеціальність — зазвичай дрібніший текст поряд
        subspecialty = ""
        if parent:
            # Шукаємо малий текст в рядку
            small = parent.find("small") or parent.find(class_=re.compile("sub|spec|small", re.I))
            if small:
                subspecialty = small.get_text(strip=True)
            else:
                # Спробуємо знайти другий рядок тексту в клітинці-батьку
                td = link.find_parent("td") or link.find_parent("div")
                if td:
                    lines = [l.strip() for l in td.get_text("\n").split("\n") if l.strip()]
                    # Перший рядок — title, далі — підспеціальність
                    if len(lines) > 1 and lines[1] != title:
                        subspecialty = lines[1]

        seen.add(full_url)
        directions.append({
            "url": full_url,
            "title": title,
            "subspecialty": subspecialty,
            "okr": okr,
        })

    log(f"  → {len(directions)} направлень знайдено")
    return directions


# ─── Крок 2: Студенти по напрямку ─────────────────────────────────────────────

def parse_students_from_html(html: str) -> list[dict]:
    """Парсить одну сторінку списку абітурієнтів"""
    soup = BeautifulSoup(html, "lxml")
    students = []

    table = soup.find("table")
    if not table:
        return []

    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Ім'я та посилання — перший <a> в рядку
        name_link = row.find("a")
        if not name_link:
            continue

        name = name_link.get_text(strip=True)
        href = name_link.get("href", "")

        # Будуємо URL для пошуку студента
        if href.startswith("/#"):
            search_url = BASE_URL + href
        elif href.startswith("#"):
            search_url = BASE_URL + "/" + href
        elif href.startswith("http"):
            search_url = href
        elif href.startswith("/"):
            search_url = BASE_URL + href
        else:
            search_url = ""

        cell_texts = [c.get_text(strip=True) for c in cells]
        full_text = " ".join(cell_texts)

        # Складові балу — це underlined числа в окремій клітинці.
        # Вони НЕ містять крапки (на відміну від загального балу 178.111).
        # Шукаємо клітинку з 2+ тризначними цілими числами БЕЗ десяткових крапок.
        scores = []
        for cell in cells:
            ct = cell.get_text(strip=True)
            if "." in ct:
                continue  # пропускаємо клітинки з дробовими числами
            nums = re.findall(r'\b(\d{3})\b', ct)
            if len(nums) >= 2:
                scores = [int(n) for n in nums]
                break

        # Якщо не знайшли через текст — спробуємо через посилання в клітинці
        # (числа можуть бути <a> або <span> елементами)
        if not scores:
            for cell in cells:
                links = cell.find_all("a")
                if len(links) >= 2:
                    nums = []
                    for a in links:
                        t = a.get_text(strip=True)
                        if re.match(r'^\d{3}$', t):
                            nums.append(int(t))
                    if len(nums) >= 2:
                        scores = nums
                        break

        # Загальний бал (формат: 191.667)
        total_score = None
        m = re.search(r'\b(\d{3}\.\d{3})\b', full_text)
        if m:
            total_score = m.group(1)

        # Статус
        status = ""
        for ct in cell_texts:
            if any(kw in ct for kw in ["наказу", "Рекомендовано", "Відхилено", "Скасовано", "Заблоковано"]):
                status = ct
                break

        # Позиція (перша клітинка — номер)
        position = cell_texts[0] if cell_texts else ""

        # Пріоритет (формат: "1 (Б)")
        priority = ""
        for ct in cell_texts:
            if re.match(r'^\d+\s*\([БМ]\)$', ct):
                priority = ct
                break

        # Квота
        quota = ""
        for ct in cell_texts:
            if "КВОТА" in ct.upper():
                quota = ct
                break

        students.append({
            "name": name,
            "search_url": search_url,
            "scores": scores,
            "total_score": total_score,
            "status": status,
            "position": position,
            "priority": priority,
            "quota": quota,
            "raw_cells": cell_texts,
        })

    return students


def get_next_page_url(html: str, current_url: str, current_page: int) -> str | None:
    """Знаходить URL наступної сторінки пагінації, або None"""
    soup = BeautifulSoup(html, "lxml")

    # Варіант 1: посилання на конкретний номер сторінки
    next_num = current_page + 1
    next_link = soup.find("a", string=str(next_num)) or \
                soup.find("a", attrs={"href": re.compile(rf"page={next_num}|/{next_num}$")})
    if next_link:
        href = next_link.get("href", "")
        if href and href != "#":
            if href.startswith("/"):
                return BASE_URL + href
            return href

    # Варіант 2: кнопка "›" або "Next"
    for selector_text in ["›", ">", "Наступна", "Next"]:
        btn = soup.find("a", string=re.compile(re.escape(selector_text)))
        if btn:
            href = btn.get("href", "")
            if href and href not in ["#", ""]:
                if href.startswith("/"):
                    return BASE_URL + href
                return href

    return None


def parse_direction_meta(html: str) -> dict:
    """
    Парсить метадані з заголовку сторінки напрямку:
    ВМ, БМmax, К, Заяв, Конкурс на бюджет, Квоти, Складові балу, форма, тривалість.
    """
    soup = BeautifulSoup(html, "lxml")
    meta = {
        "vm": "", "bm_max": "", "k": "", "zayav": "",
        "konkurs": "", "kvota1": "", "kvota2": "",
        "score_components": "", "form": "", "duration": "",
        "program": "",   # Бізнес-економіка / Економіка і великі дані / тощо
    }

    # Весь текст хедеру напрямку (зазвичай перший великий блок)
    header = soup.find(class_=re.compile("direction|header|info|title", re.I))
    if not header:
        # Fallback — беремо перший великий текстовий блок
        header = soup.find("div") or soup.body

    if not header:
        return meta

    text = header.get_text(" ", strip=True)

    # ВМ 72 • БМmax 10 • К 62
    m = re.search(r'ВМ\s+(\d+)', text)
    if m: meta["vm"] = m.group(1)

    m = re.search(r'БМmax?\s+(\d+)', text, re.I)
    if m: meta["bm_max"] = m.group(1)

    m = re.search(r'(?:^|\s)К\s+(\d+)', text)
    if m: meta["k"] = m.group(1)

    # Заяв 284 • Конкурс на бюджет 25.2
    m = re.search(r'Заяв\s+(\d+)', text)
    if m: meta["zayav"] = m.group(1)

    m = re.search(r'Конкурс на бюджет\s+([\d.]+)', text)
    if m: meta["konkurs"] = m.group(1)

    # Квота 1 1 / Квота 2 1
    m = re.search(r'Квота\s+1\s+(\d+)', text)
    if m: meta["kvota1"] = m.group(1)

    m = re.search(r'Квота\s+2\s+(\d+)', text)
    if m: meta["kvota2"] = m.group(1)

    # Форма навчання і тривалість (денна форма • 3р 10м)
    m = re.search(r'(денна|заочна|вечірня)\s+форма', text, re.I)
    if m: meta["form"] = m.group(0)

    m = re.search(r'(\d+р\s*\d*м?)', text)
    if m: meta["duration"] = m.group(1).strip()

    # Складові конкурсного балу — шукаємо посилання або текст
    score_link = soup.find("a", string=re.compile("Складові", re.I))

    # Програма/підспеціальність — остання частина хлібних крихт
    # Соц.науки / Економіка та МЕВ / Економіка / **Бізнес-економіка**
    breadcrumb_links = soup.select("a[href*='/speciality/'], a[href*='/spec/']")
    if not breadcrumb_links:
        # Шукаємо в рядку посилань що йдуть підряд (breadcrumb без class)
        breadcrumb_links = soup.select("div a, p a, span a")
    # Беремо останнє посилання що не є кнопкою і не посилання на ВНЗ
    for lnk in reversed(breadcrumb_links):
        txt = lnk.get_text(strip=True)
        if txt and len(txt) > 3 and not any(kw in txt for kw in
                ["Київська школа", "Складові", "Статистика", "КШЕ", "ВНЗ"]):
            meta["program"] = txt
            break
    if score_link:
        # Наступний елемент може містити деталі
        meta["score_components"] = score_link.get_text(strip=True)

    return meta


def fetch_direction_students(direction_url: str) -> tuple[list[dict], dict]:
    """
    Збирає всіх студентів з напрямку через requests.
    Повертає (students, meta) — список студентів і метадані напрямку.
    """
    headers = {"User-Agent": UA, "Accept-Language": "uk-UA,uk;q=0.9", "verify": "False"}
    all_students = []
    current_url = direction_url
    page_num = 1
    direction_meta = {}

    while current_url:
        log(f"    GET page {page_num}: {current_url}")
        try:
            r = requests.get(current_url, headers={"User-Agent": UA, "Accept-Language": "uk-UA,uk;q=0.9"}, timeout=20, verify=False)
            r.raise_for_status()
        except Exception as e:
            log(f"    ❌ Error fetching {current_url}: {e}")
            break

        html = r.text
        if is_rate_limited(html):
            log(f"    ⚠️  Rate limit на step 2. Чекаємо 20с...")
            time.sleep(20 + random.uniform(3, 8))
            continue

        # Парсимо метадані тільки з першої сторінки
        if page_num == 1:
            direction_meta = parse_direction_meta(html)
            log(f"    Meta: ВМ={direction_meta.get('vm')} БМmax={direction_meta.get('bm_max')} К={direction_meta.get('k')} Заяв={direction_meta.get('zayav')}")

        students = parse_students_from_html(html)
        if not students:
            log(f"    → 0 студентів, зупиняємо пагінацію")
            break

        all_students.extend(students)
        log(f"    → {len(students)} студентів на сторінці {page_num}")

        next_url = get_next_page_url(html, current_url, page_num)
        if next_url and next_url != current_url:
            current_url = next_url
            page_num += 1
            time.sleep(random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX))
        else:
            break

    return all_students, direction_meta


# ─── Крок 3: API discovery + пошук профілів ──────────────────────────────────

def load_api_cache() -> dict:
    if API_CACHE.exists():
        return json.loads(API_CACHE.read_text("utf-8"))
    return {}

def save_api_cache(data: dict):
    API_CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


async def search_via_browser_api(page, query: str, api_info: dict) -> str | None:
    """
    Викликає API прямо з браузера через fetch() — cookies і Origin виставляє
    сам Chromium, тому сервер не відхиляє запит.
    """
    raw_url  = api_info.get("raw_url", api_info.get("url", ""))
    api_url  = re.sub(r'[?&]nocache[^&]*', '', raw_url).split("?")[0].split("&")[0]

    if not api_url:
        return None

    js = """
    async ([url, query]) => {
        const allStatements = [];
        let offset = 0;
        const limit = 100;
        let total = null;

        while (true) {
            let resp;
            try {
                resp = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type':      'application/json',
                        'X-Requested-With':  'XMLHttpRequest',
                    },
                    body: JSON.stringify({
                        searchQuery: query,
                        year:        2025,
                        limit:       limit,
                        offset:      offset,
                    }),
                });
            } catch (e) {
                break;
            }

            if (!resp.ok) break;
            const data = await resp.json();

            if (total === null) total = data.count || 0;

            const statements = data.statements || data.data || data.items || [];
            if (!statements.length) break;

            allStatements.push(...statements);
            if (allStatements.length >= total) break;
            offset += limit;
        }

        return JSON.stringify({ count: total || 0, statements: allStatements });
    }
    """

    try:
        result = await page.evaluate(js, [api_url, query])
        return result
    except Exception as e:
        log(f"    Browser API error: {e}")
        return None


def parse_search_response(raw: str, target_scores: list) -> dict:
    """
    Парсить відповідь API (JSON з ключем statements) або HTML fallback.
    """
    all_rows = []
    kse_rows = []
    summary  = {}   # ← ініціалізуємо тут, до будь-яких гілок

    stripped = raw.strip()
    is_json  = stripped.startswith("{") or stripped.startswith("[")

    if is_json:
        try:
            data = json.loads(stripped)

            summary["total_apps"] = str(data.get("count", ""))
            summary["shown"]      = str(data.get("limit", ""))

            # API повертає список у ключі "statements"
            statements = (
                data.get("statements") or
                data.get("data") or
                data.get("results") or
                data.get("items") or
                (data if isinstance(data, list) else [])
            )

            for item in statements:
                if not isinstance(item, dict):
                    continue

                row_text = json.dumps(item, ensure_ascii=False)

                # ── Витягуємо складові балу ──────────────────────────
                # Поля можуть називатися scores, subjectScores, subjects, тощо
                comp_scores = []
                for key in ("scores", "subjectScores", "subjects", "marks", "components"):
                    val = item.get(key)
                    if isinstance(val, list):
                        nums = [int(v) for v in val if str(v).isdigit() and len(str(v)) == 3]
                        if len(nums) >= 2:
                            comp_scores = nums
                            break
                    elif isinstance(val, dict):
                        nums = [int(v) for v in val.values() if str(v).isdigit() and len(str(v)) == 3]
                        if len(nums) >= 2:
                            comp_scores = nums
                            break

                # Fallback: шукаємо тризначні цілі в тексті
                if not comp_scores:
                    raw_nums = re.findall(r'(?<![.\d])(\d{3})(?![.\d])', row_text)
                    if len(raw_nums) >= 2:
                        comp_scores = [int(n) for n in raw_nums]

                # ── Загальний бал ─────────────────────────────────────
                total_score = None
                for key in ("totalScore", "total", "score", "sum", "ball"):
                    val = item.get(key)
                    if val and re.match(r'^\d{3}\.\d{3}$', str(val)):
                        total_score = str(val)
                        break
                if not total_score:
                    m = re.search(r'\b(\d{3}\.\d{3})\b', row_text)
                    if m:
                        total_score = m.group(1)

                # ── Назва університету ────────────────────────────────
                university = ""
                for key in ("university", "universityName", "univer", "vuz", "vnz"):
                    if item.get(key):
                        university = str(item[key])
                        break

                parsed = {
                    "cells":       list(item.values()),
                    "headers_map": item,
                    "comp_scores": comp_scores,
                    "total_score": total_score,
                    "university":  university,
                    "raw_text":    row_text,
                }
                all_rows.append(parsed)

                # ── Матч з КШЕ ───────────────────────────────────────
                is_kse = any(kw in row_text for kw in
                             ["Київська школа", "КШЕ", "KSE", "3924"])
                scores_match = (
                    bool(target_scores) and
                    set(target_scores) <= set(comp_scores)
                )
                if is_kse or scores_match:
                    kse_rows.append(parsed)

        except (json.JSONDecodeError, Exception) as e:
            log(f"    JSON parse error: {e}")

    else:
        # HTML fallback
        soup = BeautifulSoup(raw, "lxml")
        m = re.search(r'Знайдено заяв[:\s]+(\d+)', raw)
        if m:
            summary["total_apps"] = m.group(1)

        table = soup.find("table")
        if table:
            headers = [th.get_text(strip=True) for th in table.select("thead th")]
            for row in table.select("tbody tr"):
                cells      = row.find_all("td")
                cell_texts = [c.get_text(strip=True) for c in cells]
                if not any(cell_texts):
                    continue
                row_text = " ".join(cell_texts)

                # ── Складові балу: числа без крапки ──────────────────────────
                comp_scores = []
                comp_named  = {}   # {предмет: бал}

                # Клітинка зі складовими (індекс 8 за структурою таблиці)
                # Містить: "Українська мова\n182\nМатематика\n200\nІноземна мова\n187"
                scores_cell_idx = 8  # Скл. кон. балу
                if len(cells) > scores_cell_idx:
                    sc = cells[scores_cell_idx]
                    # Витягуємо пари (назва предмета, бал)
                    sc_html = str(sc)
                    # Предмети йдуть перед числами — парсимо по рядках
                    lines = [ln.strip() for ln in sc.get_text("\n").split("\n") if ln.strip()]
                    subj, last_subj = "", ""
                    for ln in lines:
                        if re.match(r'^\d{3}$', ln):
                            score = int(ln)
                            comp_scores.append(score)
                            if last_subj:
                                comp_named[last_subj] = score
                                last_subj = ""
                        else:
                            last_subj = ln

                # Fallback: шукаємо тризначні цілі в будь-якій клітинці
                if not comp_scores:
                    for cell in cells:
                        ct = cell.get_text(strip=True)
                        if "." in ct:
                            continue
                        nums = re.findall(r'\b(\d{3})\b', ct)
                        if len(nums) >= 2:
                            comp_scores = [int(n) for n in nums]
                            break

                # ── Загальний бал ─────────────────────────────────────────────
                total_m = re.search(r'\b(\d{3}\.\d{3})\b', row_text)

                # ── Університет — посилання в клітинці ВНЗ ───────────────────
                university = ""
                if len(cells) > 9:
                    for a in cells[9].find_all("a"):
                        t = a.get_text(strip=True)
                        if t: university = t; break
                    if not university:
                        university = cells[9].get_text(strip=True)

                # ── Місця: ВМ / БМmax / К ────────────────────────────────────
                places_raw = cell_texts[5] if len(cell_texts) > 5 else ""

                # ── СБО ───────────────────────────────────────────────────────
                sbo_raw = cell_texts[7] if len(cell_texts) > 7 else ""
                sbo = sbo_raw if sbo_raw not in ("—", "-", "") else ""

                # ── Спеціальність: код + назва ────────────────────────────────
                spec_raw = cell_texts[11] if len(cell_texts) > 11 else ""
                spec_code, spec_name = "", spec_raw
                sc_m = re.match(r'^(\d{3})\s*(.*)', spec_raw)
                if sc_m:
                    spec_code = sc_m.group(1)
                    spec_name = sc_m.group(2).strip()

                # ── Д (остання колонка) ───────────────────────────────────────
                app_d = cell_texts[13] if len(cell_texts) > 13 else ""

                parsed = {
                    "cells":          cell_texts,
                    "headers_map":    dict(zip(headers, cell_texts)) if headers else {},
                    "comp_scores":    comp_scores,
                    "comp_named":     comp_named,
                    "total_score":    total_m.group(1) if total_m else None,
                    "university":     university,
                    "places_raw":     places_raw,
                    "sbo":            sbo,
                    "spec_code":      spec_code,
                    "spec_name":      spec_name,
                    "app_d":          app_d,
                    "raw_text":       row_text,
                }
                all_rows.append(parsed)
                is_kse = any(kw in row_text for kw in
                             ["Київська школа", "КШЕ", "KSE", "3924"])
                scores_match = (
                    bool(target_scores) and
                    set(target_scores) <= set(comp_scores)
                )
                if is_kse or scores_match:
                    kse_rows.append(parsed)

    return {
        "all_rows":         all_rows,
        "kse_rows":         kse_rows,
        "matched":          len(kse_rows) > 0,
        "total_apps_found": len(all_rows),
        "summary":          summary,
    }


async def discover_api(page, context, sample_query: str) -> dict:
    """
    Робить один пошук через Playwright, перехоплює ВСІ мережеві запити,
    знаходить endpoint що повертає результати пошуку.
    Повертає api_info dict або {} якщо не знайдено.
    """
    log("  🔬 API Discovery: перехоплюємо мережу під час пошуку...")

    captured = []

    async def on_request(request):
        # Тільки XHR/fetch — ігноруємо навігацію, скрипти, css
        if request.resource_type not in ("xhr", "fetch"):
            return
        url = request.url
        if any(ext in url for ext in [".js", ".css", ".png", ".ico"]):
            return
        captured.append({
            "url":     url,
            "method":  request.method,
            "headers": dict(request.headers),
            "post":    request.post_data or "",
        })

    async def on_response(response):
        url = response.url
        # Тільки XHR/fetch відповіді
        if response.request.resource_type not in ("xhr", "fetch"):
            return
        try:
            ct = response.headers.get("content-type", "")
            status = response.status
            if status == 200 and ("json" in ct or "html" in ct):
                body = await response.text()
                # Шукаємо відповідь з реальними даними (таблиця або JSON з балами)
                has_score_data = bool(re.search(r'\d{3}\.\d{3}', body))
                has_table = "<table" in body or "<tbody" in body
                has_search_info = "Знайдено заяв" in body or "Показано" in body
                if has_score_data or (has_table and has_search_info):
                    for req in captured:
                        if req["url"] == url:
                            req["response_ct"] = ct
                            req["response_preview"] = body[:500]
                            req["has_results"] = True
                            break
        except:
            pass

    page.on("request", on_request)
    page.on("response", on_response)

    await context.clear_cookies()
    try:
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(1)
    except:
        pass

    # Заповнюємо форму
    search_input = await page.query_selector(
        "input[type='text'], input[type='search'], input[name*='search'], "
        "input[placeholder*='ошук'], input.search, #search-input, input[value]"
    )
    if search_input:
        await search_input.click()
        await page.keyboard.press("Control+a")
        await search_input.fill(sample_query)
        await asyncio.sleep(0.3)
        search_btn = await page.query_selector(
            "button:has-text('ПОШУК'), button[type='submit'], "
            "input[type='submit'], button.search-btn"
        )
        if search_btn:
            await search_btn.click()
        else:
            await search_input.press("Enter")
    else:
        # Якщо форми нема — йдемо по hash URL
        hash_url = f"{BASE_URL}/#search-{sample_query.replace(' ', '+')}"
        await page.goto(hash_url, wait_until="networkidle", timeout=20000)

    try:
        await page.wait_for_selector("table tbody tr", timeout=15000)
    except:
        await asyncio.sleep(5)

    page.remove_listener("request", on_request)
    page.remove_listener("response", on_response)

    # Аналізуємо перехоплені запити
    api_candidates = [r for r in captured if r.get("has_results")]
    log(f"  🔬 Перехоплено {len(captured)} запитів, {len(api_candidates)} з результатами")

    if not api_candidates:
        log("  ⚠️  API не знайдено — будемо використовувати Playwright для кожного студента")
        return {}

    best = api_candidates[0]
    log(f"  ✅ API знайдено: {best['method']} {best['url']}")
    log(f"     Preview: {best.get('response_preview', '')[:100]}")

    # Визначаємо тип параметра
    api_url = best["url"]
    method  = best["method"]

    # Перевіряємо чи query є в URL
    encoded_q = requests.utils.quote(sample_query)
    if sample_query in api_url or encoded_q in api_url:
        # query в path або query string
        base_api_url = re.sub(re.escape(encoded_q) + r'[^&]*', "{QUERY}", api_url)
        base_api_url = re.sub(re.escape(sample_query) + r'[^&]*', "{QUERY}", base_api_url)
        param_type = "path" if "?" not in base_api_url else "query"
        param_name = ""
        if "?" in api_url:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(api_url)
            qs = parse_qs(parsed.query)
            for k, v in qs.items():
                if sample_query in " ".join(v) or encoded_q in " ".join(v):
                    param_name = k
                    break
            param_type = "query"
    else:
        param_type = "body" if method == "POST" else "query"
        param_name = "query"
        base_api_url = api_url

    api_info = {
        "url":          re.sub(r'[?&]nocache[^&]*', '', api_url).rstrip("/?&"),
        "raw_url":      api_url,
        "method":       method,
        "param_type":   param_type,
        "param_name":   param_name,
        "extra_headers": {k: v for k, v in best["headers"].items()
                          if k.lower() in ["x-requested-with", "x-csrf-token",
                                           "authorization", "content-type"]},
        "response_ct":  best.get("response_ct", ""),
    }

    # Зберігаємо cookies з Playwright-сесії — без них API повертає порожньо
    try:
        pw_cookies = await context.cookies()
        api_info["cookies"] = {c["name"]: c["value"] for c in pw_cookies}
        log(f"     Cookies збережено: {list(api_info['cookies'].keys())}")
    except Exception as e:
        log(f"     Cookies не вдалося зберегти: {e}")
        api_info["cookies"] = {}

    save_api_cache(api_info)
    log(f"  💾 API info збережено в {API_CACHE}")
    return api_info

def extract_search_query(search_url: str) -> str:
    """Витягує пошуковий запит з hash URL типу /#search-3924-9757994+Г.+А.+2025"""
    if "#search-" in search_url:
        raw = search_url.split("#search-")[1]
        return unquote(raw.replace("+", " ")).strip()
    return ""


async def scrape_student_profile(page, context, student: dict, api_info: dict) -> dict | None:
    """
    Спершу пробує прямий API (швидко ~2с), якщо не вийшло — Playwright форма (15с+).
    """
    search_url    = student.get("search_url", "")
    target_scores = student.get("scores", [])

    if not search_url:
        return None

    query = extract_search_query(search_url)
    if not query:
        log(f"    Не вдалося витягти query з {search_url}")
        return None

    raw_response = None
    used_api     = False

    # ── Спроба 1: API через браузер (fetch з Chromium) ───────
    if api_info:
        delay = random.uniform(API_DELAY_MIN, API_DELAY_MAX)
        await asyncio.sleep(delay)

        # Переконуємось що сторінка на домені сайту (для same-origin fetch)
        current_url = page.url
        if not current_url.startswith(BASE_URL):
            try:
                await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=12000)
            except Exception:
                pass

        raw_response = await search_via_browser_api(page, query, api_info)
        if raw_response and not is_rate_limited(raw_response):
            try:
                _check = json.loads(raw_response)
                if _check.get("count", 0) > 0:
                    used_api = True
                    log(f"    [API {delay:.1f}s] {query[:45]}")
                else:
                    raw_response = None
            except Exception:
                raw_response = None
        else:
            raw_response = None

    # ── Спроба 2: Playwright + форма ─────────────────────────
    if not raw_response:
        delay = random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX)
        log(f"    [PW {delay:.0f}s] {query[:45]}")
        await asyncio.sleep(delay)
        await context.clear_cookies()

        for attempt in range(2):
            try:
                await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(0.8)
                break
            except PWTimeout:
                if attempt == 1:
                    return None
                await asyncio.sleep(5)

        search_input = await page.query_selector(
            "input[type='text'], input[type='search'], "
            "input[name*='search'], input[placeholder*='ошук'], "
            "input.search, #search-input"
        )
        if search_input:
            await search_input.click()
            await page.keyboard.press("Control+a")
            await search_input.fill(query)
            await asyncio.sleep(0.3)
            btn = await page.query_selector(
                "button:has-text('ПОШУК'), button[type='submit'], "
                "input[type='submit'], button.search-btn"
            )
            if btn:
                await btn.click()
            else:
                await search_input.press("Enter")
        else:
            await page.goto(search_url, wait_until="networkidle", timeout=25000)

        try:
            await page.wait_for_selector("table tbody tr", timeout=18000)
        except PWTimeout:
            content = await page.content()
            if is_rate_limited(content):
                log(f"    ⚠️  Rate limit +35с")
                await asyncio.sleep(35 + random.uniform(5, 10))
                await context.clear_cookies()
                try:
                    await page.goto(search_url, wait_until="networkidle", timeout=25000)
                    await page.wait_for_selector("table tbody tr", timeout=15000)
                except:
                    pass

        raw_response = await page.content()

    if not raw_response:
        return None

    result = parse_search_response(raw_response, target_scores)
    icon = "✅" if result["matched"] else "⚠️ "
    mode = "API" if used_api else "PW"
    log(f"    {icon}[{mode}] {student['name'][:32]}: {result['total_apps_found']} заяв, {len(result['kse_rows'])} КШЕ")

    return {
        "name":          student["name"],
        "search_url":    search_url,
        "query":         query,
        "target_scores": target_scores,
        **result,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(workers: int = 1):
    log("=" * 60)
    log("🚀 KSE Applicants Scraper запущено")
    log("=" * 60)

    progress = load_progress()

    # ═══════════════════════════════════════════════════════════
    # КРОК 1: Конкурсні пропозиції
    # ═══════════════════════════════════════════════════════════
    if not progress["directions"]:
        log("\n📋 КРОК 1: Збираємо конкурсні пропозиції...")
        directions = fetch_directions()

        if not directions:
            log("❌ Не знайдено жодного напрямку! Перевір HTML структуру.")
            log("   Запусти з --debug щоб побачити сирий HTML")
            sys.exit(1)

        log(f"✅ Знайдено {len(directions)} напрямків:")
        for d in directions:
            log(f"   → {d['url']}")
            log(f"     {d['title']} / {d['subspecialty']}")

        progress["directions"] = directions
        save_progress(progress)
    else:
        directions = progress["directions"]
        log(f"\n📋 КРОК 1: Завантажено {len(directions)} напрямків з кешу")

    # ═══════════════════════════════════════════════════════════
    # КРОК 2: Студенти по кожному напрямку
    # ═══════════════════════════════════════════════════════════
    log(f"\n👥 КРОК 2: Студенти з {len(directions)} напрямків...")

    for direction in directions:
        dir_url = direction["url"]

        if dir_url in progress["done_dirs"]:
            n = len(progress["students"].get(dir_url, {}).get("list", []))
            log(f"  ⏭️  {direction['title'][:55]}: {n} студентів (кеш)")
            continue

        log(f"\n  📄 {direction['title'][:60]}")
        log(f"     {direction['subspecialty']}")

        students, dir_meta = fetch_direction_students(dir_url)

        progress["students"][dir_url] = {
            "direction": direction,
            "meta": dir_meta,
            "list": students,
        }
        progress["done_dirs"].append(dir_url)
        save_progress(progress)

        log(f"  ✅ {len(students)} студентів всього")
        time.sleep(random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX))

    # Зберігаємо kse_students.csv після кроку 2
    csv_write_students(progress)

    # ═══════════════════════════════════════════════════════════
    # КРОК 3: Профіль кожного студента (Playwright)
    # ═══════════════════════════════════════════════════════════

    # Дедуплікуємо студентів по search_url, зберігаємо з яким direction вони пов'язані
    unique_students = {}
    for dir_url, dir_data in progress["students"].items():
        for student in dir_data.get("list", []):
            su = student.get("search_url", "")
            if su and su not in unique_students:
                unique_students[su] = {**student, "_dir_url": dir_url}

    total_students = len(unique_students)

    # Resume: читаємо вже оброблені search_url з CSV
    csv_ensure_profiles_header()
    done_profiles = csv_get_done_profiles()
    remaining = total_students - len(done_profiles)

    log(f"\n🔍 КРОК 3: Профілі для {total_students} унікальних студентів")
    log(f"   Вже в CSV: {len(done_profiles)}, залишилося: {remaining}")
    log(f"   Воркерів: {workers}")
    log(f"   Затримка між пошуками: {SEARCH_DELAY_MIN}-{SEARCH_DELAY_MAX}с (rate limit, per worker)")
    if remaining > 0:
        effective_delay = SEARCH_DELAY_MAX / workers
        log(f"   Приблизний час: ~{int(remaining * effective_delay / 60)} хвилин при {workers} воркерах")

    # Черга студентів для обробки
    queue: asyncio.Queue = asyncio.Queue()
    for su, student in unique_students.items():
        if su not in done_profiles:
            await queue.put((su, student))

    # CSV lock — щоб воркери не писали одночасно
    csv_lock = asyncio.Lock()
    # Shared state
    shared = {
        "done_count": len(done_profiles),
        "api_info":   None,
        "api_zeros":  0,
    }

    async def worker(worker_id: int, browser):
        """Один воркер: бере студентів з черги, парсить, пише в CSV."""
        context = await browser.new_context(
            user_agent=UA,
            locale="uk-UA",
            viewport={"width": 1280, "height": 900},
            extra_http_headers={"Accept-Language": "uk-UA,uk;q=0.9"},
        )
        page = await context.new_page()
        await page.route(
            "**/*.{png,jpg,jpeg,gif,svg,woff,woff2}",
            lambda route: route.abort()
        )

        while True:
            try:
                su, student = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            dir_data  = progress["students"][student["_dir_url"]]
            direction = {**dir_data["direction"], "meta": dir_data.get("meta", {})}
            api_info  = shared["api_info"]

            try:
                profile = await scrape_student_profile(page, context, student, api_info)
            except Exception as e:
                log(f"  [W{worker_id}] ❌ {student['name'][:30]}: {e}")
                profile = {"error": str(e), "name": student["name"]}

            # Автоінвалідація API (спільна для всіх воркерів)
            if api_info and profile and not profile.get("error"):
                if profile.get("total_apps_found", 0) == 0:
                    shared["api_zeros"] += 1
                    if shared["api_zeros"] >= 3:
                        log(f"  ⚠️  API дає 0 результатів — скидаємо на Playwright")
                        shared["api_info"] = {}
                        API_CACHE.unlink(missing_ok=True)
                        shared["api_zeros"] = 0
                else:
                    shared["api_zeros"] = 0

            async with csv_lock:
                csv_append_profile(student, direction, profile)
                shared["done_count"] += 1
                done_count = shared["done_count"]
                left = total_students - done_count
                cur_api = shared["api_info"]
                if cur_api:
                    eta = f"~{int(left * API_DELAY_MAX / workers / 60)}хв"
                else:
                    eta = f"~{int(left * SEARCH_DELAY_MAX / workers / 60)}хв"
                log(f"  [W{worker_id}] 💾 {done_count}/{total_students} | left {eta}")

            queue.task_done()

        await page.close()
        await context.close()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
            ],
        )

        # ── API Discovery (один раз, воркер 0) ────────────────────
        api_info = load_api_cache()
        if api_info:
            log(f"  🔗 API з кешу: {api_info.get('method')} {api_info.get('raw_url', api_info.get('url'))}")
        elif remaining > 0:
            disc_context = await browser.new_context(user_agent=UA)
            disc_page    = await disc_context.new_page()
            sample_student = next(
                (s for su, s in unique_students.items() if su not in done_profiles),
                None
            )
            if sample_student:
                sample_q = extract_search_query(sample_student.get("search_url", ""))
                if sample_q:
                    api_info = await discover_api(disc_page, disc_context, sample_q)
            await disc_page.close()
            await disc_context.close()

        shared["api_info"] = api_info

        if api_info:
            est = int(remaining * API_DELAY_MAX / workers / 60)
            log(f"  🚀 API mode: ~{est} хв при {workers} воркерах")
        else:
            est = int(remaining * SEARCH_DELAY_MAX / workers / 60)
            log(f"  🐢 Playwright mode: ~{est} хв при {workers} воркерах")

        # ── Запускаємо N воркерів паралельно ──────────────────────
        await asyncio.gather(*[worker(i + 1, browser) for i in range(workers)])

        await browser.close()

    # ═══════════════════════════════════════════════════════════
    # Фінальна статистика (читаємо з CSV)
    # ═══════════════════════════════════════════════════════════
    total_rows = 0
    matched = errors = no_profile = 0

    if CSV_PROFILES.exists():
        with open(CSV_PROFILES, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total_rows += 1
                pm = row.get("profile_matched", "")
                if pm == "YES":
                    matched += 1
                elif pm == "ERROR":
                    errors += 1
                elif pm == "NO_PROFILE":
                    no_profile += 1

    # Оновлюємо kse_students.csv після завершення всіх кроків
    csv_write_students(progress)

    log("\n" + "=" * 60)
    log(f"✅ Готово!")
    log(f"   {CSV_PROFILES}: {total_rows} рядків")
    log(f"   Matched (КШЕ рядок знайдено): {matched}")
    log(f"   Помилок:                       {errors}")
    log(f"   Без профілю:                   {no_profile}")
    log(f"   {CSV_STUDENTS}: список студентів по напрямках")
    log("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KSE abit-poisk scraper")
    parser.add_argument(
        "workers",
        nargs="?",
        type=int,
        default=1,
        help="Кількість паралельних воркерів для кроку 3 (default: 1). "
             "Приклад: python kse_scrape.py 4",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Показати сирий HTML сторінки університету і вийти",
    )
    args = parser.parse_args()

    if args.debug:
        r = requests.get(UNIVER_URL, headers={"User-Agent": UA}, timeout=20, verify=False)
        print(r.text[:5000])
        sys.exit(0)

    if args.workers < 1:
        print("workers має бути >= 1")
        sys.exit(1)
    if args.workers > 8:
        print(f"⚠️  {args.workers} воркерів — це багато, сайт може заблокувати. Рекомендовано <= 4.")

    asyncio.run(main(workers=args.workers))