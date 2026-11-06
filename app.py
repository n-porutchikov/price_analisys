import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import tempfile
import os
import io

# ==============================================================================
# 1. КОНСТАНТЫ И ЛОГИКА АНАЛИЗА (Взято из твоего exe-файла без изменений)
# ==============================================================================
IQR_MULTIPLIER = 3.0
MIN_IQR_SAMPLE = 8
MIN_JUMP_THRESHOLD_PCT = 10.0
MIN_DEVIATION_THRESHOLD_PCT = 15.0
MIN_HISTORY_DATES = 5
EXCLUDE_PRICE_ANOMALIES = True
PRICE_INDEX_HIGH = 1.05
PRICE_INDEX_LOW = 0.95
DISCOUNT_DIFF_PP = 5.0
MIN_SKU_FOR_CATEGORY_RANKING = 10
EXCEL_DATA_ROWS_PER_SHEET = 1_048_575

ALIASES = {
    "city": ["Город", "Населенный пункт", "Населённый пункт", "Регион", "City"],
    "sku": ["СКЮ", "SKU", "Артикул", "Код товара", "Код SKU", "ID товара", "Номенклатура"],
    "competitor": ["Конкурент", "Название конкурента", "Сеть конкурента", "Ритейлер", "Сеть", "Competitor"],
    "category": ["Категория", "Категория товара", "Category"],
    "subcategory": ["Подкатегория", "Подкатегория товара", "Субкатегория", "Subcategory"],
    "group": ["Группа", "Товарная группа", "Группа товаров", "Group"],
    "date": ["Обновление", "Дата обновления", "Дата", "Дата замера", "Дата сбора", "Дата мониторинга", "Date"],
    "competitor_regular": ["Регулярная цена конкурента", "Цена конкурента", "Цена конкурента регулярная",
                           "Цена регулярная конкурента", "Рег. цена конкурента", "Цена конкурента без скидки"],
    "lenta_regular": ["Регулярная цена Лента", "Регулярная цена Ленты", "Регулярная цена ленты",
                      "Цена Ленты", "Цена ленты регулярная", "Цена регулярная Лента",
                      "Рег. цена Ленты", "Цена Ленты без скидки"],
    "turnover": ["Товарооборот", "ТО", "Оборот", "Выручка", "Товарооборот, руб.", "Turnover"],
    "competitor_promo": ["Акционная цена конкурента", "Промо цена конкурента", "Цена конкурента по акции",
                         "Цена конкурента акционная", "Цена по акции конкурента"],
    "lenta_promo": ["Акционная цена Лента", "Акционная цена Ленты", "Акционная цена ленты",
                    "Промо цена Ленты", "Цена Ленты по акции", "Цена ленты акционная", "Цена по акции Ленты"],
}
DISPLAY = {
    "city": "Город", "sku": "СКЮ", "competitor": "Конкурент",
    "category": "Категория", "subcategory": "Подкатегория", "group": "Группа",
    "date": "Обновление", "competitor_regular": "Регулярная цена конкурента",
    "lenta_regular": "Регулярная цена Ленты", "turnover": "Товарооборот",
    "competitor_promo": "Акционная цена конкурента", "lenta_promo": "Акционная цена Ленты",
}
REQUIRED = ["sku", "competitor", "date", "competitor_regular", "lenta_regular", "turnover"]
OPTIONAL_TEXT = ["city", "category", "subcategory", "group"]

def normalize_header(value):
    value = str(value).strip().lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "", value)

def resolve_columns(columns):
    lookup = {}
    for col in columns:
        lookup.setdefault(normalize_header(col), str(col))
    result, missing = {}, []
    for canonical, aliases in ALIASES.items():
        found = next((lookup[normalize_header(a)] for a in aliases
                      if normalize_header(a) in lookup), None)
        if found is not None:
            result[canonical] = found
        elif canonical in REQUIRED:
            missing.append(f"{DISPLAY.get(canonical, canonical)}: {', '.join(aliases)}")
    if missing:
        raise ValueError("Не найдены обязательные столбцы:\n- " + "\n- ".join(missing))
    return result

def parse_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "-": pd.NA, "—": pd.NA, "нет": pd.NA})
    s = s.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    both = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    comma_decimal = both & (s.str.rfind(",") > s.str.rfind("."))
    dot_decimal = both & ~comma_decimal
    s.loc[comma_decimal] = (s.loc[comma_decimal].str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    s.loc[dot_decimal] = s.loc[dot_decimal].str.replace(",", "", regex=False)
    only_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def parse_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

def clean_string(series):
    return series.astype("string").str.strip().replace("", pd.NA)

def safe_ratio(a, b):
    result = a / b
    return result.where(a.notna() & b.notna() & b.ne(0))

def pct_change(series):
    previous = series.shift(1)
    result = (series / previous - 1) * 100
    return result.where(previous.gt(0) & series.gt(0))

def iqr_threshold(series, minimum):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < MIN_IQR_SAMPLE:
        return float(minimum)
    q1, q3 = values.quantile([0.25, 0.75])
    threshold = q3 + IQR_MULTIPLIER * (q3 - q1)
    return float(max(minimum, threshold)) if np.isfinite(threshold) else float(minimum)

def reliability_label(n):
    if pd.isna(n): return "Нет данных"
    n = int(n)
    if n >= 3: return "Высокая (≥3 конкурента)"
    if n == 2: return "Средняя (2 конкурента)"
    if n == 1: return "Низкая (1 конкурент)"
    return "Нет данных"

# ==============================================================================
# 2. ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА (Возвращает готовый Excel в виде байтов)
# ==============================================================================
def run_analysis(input_file, sheet_index):
    try:
        raw = pd.read_excel(input_file, sheet_name=sheet_index, engine="openpyxl")
    except ValueError:
        raw = pd.read_excel(input_file, sheet_name=0, engine="openpyxl")
        
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Выбранный лист пуст или не является таблицей.")

    colmap = resolve_columns(raw.columns)
    df = raw.rename(columns={actual: key for key, actual in colmap.items()}).copy()

    for col in OPTIONAL_TEXT:
        if col not in df.columns:
            df[col] = "Не указано"
        else:
            df[col] = clean_string(df[col]).fillna("Не указано")
    for col in ["sku", "competitor"]:
        df[col] = clean_string(df[col])
    df["date"] = parse_dates(df["date"])
    for col in ["competitor_regular", "lenta_regular", "turnover"]:
        df[col] = parse_numeric(df[col])
    for col in ["competitor_promo", "lenta_promo"]:
        df[col] = parse_numeric(df[col]) if col in df.columns else np.nan

    missing_key = df[["sku", "competitor", "date"]].isna().any(axis=1)
    bad_price = (df["competitor_regular"].isna() | df["lenta_regular"].isna() | df["competitor_regular"].le(0) | df["lenta_regular"].le(0))
    bad_turnover = df["turnover"].isna() | df["turnover"].lt(0)
    invalid_mask = missing_key | bad_price | bad_turnover
    invalid = df.loc[invalid_mask].copy()
    
    valid = df.loc[~invalid_mask].copy()
    dup_mask = valid.duplicated(keep="first")
    duplicates = valid.loc[dup_mask].copy()
    valid = valid.loc[~dup_mask].copy()

    conflict_key = ["city", "sku", "competitor", "date"]
    value_cols = ["competitor_regular", "lenta_regular", "turnover", "competitor_promo", "lenta_promo"]
    variants = (valid.groupby(conflict_key, dropna=False)[value_cols].apply(lambda x: len(x.drop_duplicates())).rename("Вариантов").reset_index())
    conflict_keys = variants.loc[variants["Вариантов"] > 1, conflict_key]
    
    if not conflict_keys.empty:
        conflict_rows = valid.merge(conflict_keys, on=conflict_key, how="inner")
        valid = valid.merge(conflict_keys.assign(_c=True), on=conflict_key, how="left")
        valid = valid.loc[valid["_c"].isna()].drop(columns="_c")
    else:
        conflict_rows = valid.iloc[0:0].copy()

    valid = valid.sort_values(["city", "sku", "competitor", "date"], kind="mergesort").reset_index(drop=True)

    series_key = ["city", "sku", "competitor"]
    valid["Изменение цены конкурента, %"] = valid.groupby(series_key)["competitor_regular"].transform(pct_change)
    valid["Изменение цены Ленты, %"] = valid.groupby(series_key)["lenta_regular"].transform(pct_change)

    jump_limits = (valid.groupby("group")["Изменение цены конкурента, %"].apply(lambda x: iqr_threshold(x.abs(), MIN_JUMP_THRESHOLD_PCT)).to_dict())
    valid["Порог скачка, %"] = valid["group"].map(jump_limits).fillna(MIN_JUMP_THRESHOLD_PCT)

    valid["Медианная цена конкурента в ряду"] = valid.groupby(series_key)["competitor_regular"].transform("median")
    valid["Отклонение от медианы ряда, %"] = ((valid["competitor_regular"] / valid["Медианная цена конкурента в ряду"] - 1) * 100)

    deviation_limits = (valid.groupby("group")["Отклонение от медианы ряда, %"].apply(lambda x: iqr_threshold(x.abs(), MIN_DEVIATION_THRESHOLD_PCT)).to_dict())
    valid["Порог отклонения, %"] = valid["group"].map(deviation_limits).fillna(MIN_DEVIATION_THRESHOLD_PCT)
    valid["Дат в истории"] = valid.groupby(series_key)["date"].transform("nunique")

    jump_flag = (valid["Изменение цены конкурента, %"].abs().gt(valid["Порог скачка, %"]) & valid["Изменение цены Ленты, %"].abs().le(valid["Порог скачка, %"]))
    deviation_flag = (valid["Отклонение от медианы ряда, %"].abs().gt(valid["Порог отклонения, %"]) & valid["Дат в истории"].ge(MIN_HISTORY_DATES))

    valid["Ценовая аномалия"] = np.select([jump_flag & deviation_flag, jump_flag, deviation_flag], ["Скачок цены и отклонение", "Резкий скачок", "Отклонение от медианы"], default="")
    
    anomalies = valid.loc[valid["Ценовая аномалия"].ne("")].copy()
    analysis_rows = valid.loc[valid["Ценовая аномалия"].eq("")].copy() if EXCLUDE_PRICE_ANOMALIES else valid.copy()

    last_dates = (analysis_rows.groupby(["city", "sku"])["date"].max().rename("last_date").reset_index())
    latest_rows = analysis_rows.merge(last_dates, on=["city", "sku"], how="inner")
    latest_rows = latest_rows.loc[latest_rows["date"].eq(latest_rows["last_date"])].copy()

    latest_agg = latest_rows.groupby(["city", "sku"], as_index=False).agg(
        **{"category": ("category", "first"), "subcategory": ("subcategory", "first"), "group": ("group", "first"),
           "Цена Ленты (рег.)": ("lenta_regular", "median"), "Цена рынка (медиана по конкурентам)": ("competitor_regular", "median"),
           "Промо Ленты": ("lenta_promo", "median"), "Промо рынка (медиана)": ("competitor_promo", "median"),
           "Товарооборот": ("turnover", "median"), "Количество конкурентов": ("competitor", "nunique"), 
           "Последняя дата наблюдения": ("date", "max")})

    latest_agg["Индекс регулярной цены"] = safe_ratio(latest_agg["Цена Ленты (рег.)"], latest_agg["Цена рынка (медиана по конкурентам)"])
    
    latest_agg["Скидка Ленты, %"] = ((1 - latest_agg["Промо Ленты"] / latest_agg["Цена Ленты (рег.)"]) * 100).where(latest_agg["Промо Ленты"].gt(0) & latest_agg["Промо Ленты"].le(latest_agg["Цена Ленты (рег.)"]))
    latest_agg["Скидка конкурента, %"] = ((1 - latest_agg["Промо рынка (медиана)"] / latest_agg["Цена рынка (медиана по конкурентам)"]) * 100).where(latest_agg["Промо рынка (медиана)"].gt(0) & latest_agg["Промо рынка (медиана)"].le(latest_agg["Цена рынка (медиана по конкурентам)"]))
    latest_agg["Разница скидок, п.п."] = latest_agg["Скидка Ленты, %"] - latest_agg["Скидка конкурента, %"]

    sku = latest_agg.copy()
    sku["Медианный ТО подкатегории"] = sku.groupby(["city", "subcategory"])["Товарооборот"].transform("median")
    high_price = sku["Индекс регулярной цены"].gt(PRICE_INDEX_HIGH)
    low_price = sku["Индекс регулярной цены"].lt(PRICE_INDEX_LOW)
    high_turnover = sku["Товарооборот"].gt(sku["Медианный ТО подкатегории"])

    sku["Сегмент"] = np.select([high_price & high_turnover, high_price & ~high_turnover, low_price & high_turnover, low_price & ~high_turnover], ["Дороже рынка / высокий ТО", "Дороже рынка / низкий ТО", "Дешевле рынка / высокий ТО", "Дешевле рынка / низкий ТО"], default="Ценовой паритет")
    no_data = (sku["Индекс регулярной цены"].isna() | sku["Товарооборот"].isna() | sku["Медианный ТО подкатегории"].isna())
    sku.loc[no_data, "Сегмент"] = "Недостаточно данных"

    diff = sku["Разница скидок, п.п."]
    sku["Сравнение скидок"] = np.select([diff.gt(DISCOUNT_DIFF_PP), diff.lt(-DISCOUNT_DIFF_PP), diff.notna()], ["Скидка Ленты глубже", "Скидка Ленты меньше", "Скидки сопоставимы"], default="Нет данных")

    category = sku.groupby(["city", "category"], as_index=False, dropna=False).agg(**{"Товарооборот": ("Товарооборот", "sum"), "Количество SKU": ("sku", "nunique"), "Медианный индекс цены": ("Индекс регулярной цены", "median"), "Дороже рынка / низкий ТО": ("Сегмент", lambda x: int((x == "Дороже рынка / низкий ТО").sum())), "Дешевле рынка / высокий ТО": ("Сегмент", lambda x: int((x == "Дешевле рынка / высокий ТО").sum()))})
    city_total = category.groupby("city")["Товарооборот"].transform("sum")
    category["Доля ТО в городе, %"] = safe_ratio(category["Товарооборот"], city_total) * 100

    risk_categories = (category.loc[category["Количество SKU"] >= MIN_SKU_FOR_CATEGORY_RANKING].sort_values(["Дороже рынка / низкий ТО", "Товарооборот"], ascending=[False, False]).head(10))
    opportunity_categories = (category.loc[category["Количество SKU"] >= MIN_SKU_FOR_CATEGORY_RANKING].sort_values(["Дешевле рынка / высокий ТО", "Товарооборот"], ascending=[False, False]).head(10))

    sku_cols = ["city", "sku", "category", "subcategory", "Товарооборот", "Индекс регулярной цены", "Сегмент", "Сравнение скидок", "Количество конкурентов"]
    risk_sku = (sku.loc[sku["Сегмент"] == "Дороже рынка / низкий ТО"].sort_values(["Количество конкурентов", "Товарооборот"], ascending=[False, False]).head(15)[sku_cols])
    opportunity_sku = (sku.loc[sku["Сегмент"] == "Дешевле рынка / высокий ТО"].sort_values(["Количество конкурентов", "Товарооборот"], ascending=[False, False]).head(15)[sku_cols])

    excluded_parts = [x for x in [invalid, duplicates, conflict_rows, anomalies] if not x.empty]
    excluded = pd.concat(excluded_parts, ignore_index=True, sort=False) if excluded_parts else pd.DataFrame()

    # Экспортные формы
    clean_export = sku.rename(columns={"city": "Город", "sku": "СКЮ", "category": "Категория", "subcategory": "Подкатегория", "group": "Группа"})
    final_cols = ["Сегмент", "Сравнение скидок", "Город", "СКЮ", "Категория", "Подкатегория", "Товарооборот", "Индекс регулярной цены", "Скидка Ленты, %", "Скидка конкурента, %", "Разница скидок, п.п.", "Количество конкурентов", "Последняя дата наблюдения"]
    clean_export = clean_export[[c for c in final_cols if c in clean_export.columns]]

    # Запись в Excel (в буфер обмена)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#39785D", "text_wrap": True, "valign": "vcenter"})
        
        clean_export.to_excel(writer, sheet_name="1. Очищенные данные", index=False)
        ws = writer.sheets["1. Очищенные данные"]
        ws.freeze_panes(1, 0)
        
        if "Индекс регулярной цены" in clean_export.columns:
            c = clean_export.columns.get_loc("Индекс регулярной цены")
            ws.conditional_format(1, c, len(clean_export), c, {"type": "3_color_scale"})

    output.seek(0)
    return output

# ==============================================================================
# 3. ИНТЕРФЕЙС STREAMLIT (Веб-страница)
# ==============================================================================
st.set_page_config(page_title="Анализ цен Ленты", page_icon="🛒", layout="centered")

st.title("🛒 Анализ цен и товарооборота «Ленты»")
st.markdown("Загрузите исходный Excel-файл, и программа сформирует готовый аналитический отчет.")

uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.info("✅ Файл успешно загружен! Нажмите кнопку ниже, чтобы запустить обработку.")
    
    if st.button("🚀 Запустить анализ", use_container_width=True):
        with st.spinner("⏳ Идет обработка данных, это может занять несколько минут..."):
            try:
                # Вызов нашей функции
                excel_data = run_analysis(uploaded_file, sheet_index=1)
                
                st.success("🎉 Анализ успешно завершен!")
                st.balloons()
                
                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать готовый отчет",
                    data=excel_data,
                    file_name="Итоговый_отчет_Лента.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Произошла ошибка при обработке: {e}")

st.markdown("---")
st.caption("Разработано для анализа цен.")
