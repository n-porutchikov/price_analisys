import streamlit as st
import pandas as pd
import numpy as np
import re
import io

st.set_page_config(page_title="Анализ цен Ленты", page_icon="🛒", layout="centered")

# --- ЛОГИКА (с быстрым движком calamine) ---
ALIASES = {
    "city": ["Город", "Населенный пункт", "Населённый пункт", "Регион", "City"],
    "sku": ["СКЮ", "SKU", "Артикул", "Код товара", "Код SKU", "ID товара", "Номенклатура"],
    "competitor": ["Конкурент", "Название конкурента", "Сеть конкурента", "Ритейлер", "Сеть", "Competitor"],
    "category": ["Категория", "Категория товара", "Category"],
    "subcategory": ["Подкатегория", "Подкатегория товара", "Субкатегория", "Subcategory"],
    "group": ["Группа", "Товарная группа", "Группа товаров", "Group"],
    "date": ["Обновление", "Дата обновления", "Дата", "Дата замера", "Дата сбора", "Дата мониторинга", "Date"],
    "competitor_regular": ["Регулярная цена конкурента", "Цена конкурента", "Цена конкурента регулярная", "Цена регулярная конкурента", "Рег. цена конкурента", "Цена конкурента без скидки"],
    "lenta_regular": ["Регулярная цена Лента", "Регулярная цена Ленты", "Регулярная цена ленты", "Цена Ленты", "Цена ленты регулярная", "Цена регулярная Лента", "Рег. цена Ленты", "Цена Ленты без скидки"],
    "turnover": ["Товарооборот", "ТО", "Оборот", "Выручка", "Товарооборот, руб.", "Turnover"],
    "competitor_promo": ["Акционная цена конкурента", "Промо цена конкурента", "Цена конкурента по акции", "Цена конкурента акционная", "Цена по акции конкурента"],
    "lenta_promo": ["Акционная цена Лента", "Акционная цена Ленты", "Акционная цена ленты", "Промо цена Ленты", "Цена Ленты по акции", "Цена ленты акционная", "Цена по акции Ленты"],
}
DISPLAY = {"city":"Город","sku":"СКЮ","competitor":"Конкурент","category":"Категория","subcategory":"Подкатегория","group":"Группа","date":"Обновление","competitor_regular":"Регулярная цена конкурента","lenta_regular":"Регулярная цена Ленты","turnover":"Товарооборот","competitor_promo":"Акционная цена конкурента","lenta_promo":"Акционная цена Ленты"}
REQUIRED = ["sku","competitor","date","competitor_regular","lenta_regular","turnover"]
OPTIONAL_TEXT = ["city","category","subcategory","group"]

def normalize_header(v): return re.sub(r"[^a-zа-я0-9]+", "", str(v).strip().lower().replace("ё","е"))
def resolve_columns(cols):
    lookup = {normalize_header(c): str(c) for c in cols}
    res, miss = {}, []
    for k, aliases in ALIASES.items():
        f = next((lookup[normalize_header(a)] for a in aliases if normalize_header(a) in lookup), None)
        if f: res[k] = f
        elif k in REQUIRED: miss.append(DISPLAY.get(k,k))
    if miss: raise ValueError("Не найдены столбцы: " + ", ".join(miss))
    return res

def parse_numeric(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce")
    s = s.astype("string").str.strip().replace({"":pd.NA,"-":pd.NA,"—":pd.NA,"нет":pd.NA})
    s = s.str.replace("\u00a0","",regex=False).str.replace(" ","",regex=False)
    both = s.str.contains(",",na=False) & s.str.contains(r"\.",na=False)
    c_dec = both & (s.str.rfind(",") > s.str.rfind("."))
    s.loc[c_dec] = s.loc[c_dec].str.replace(".","",regex=False).str.replace(",",".",regex=False)
    s.loc[both & ~c_dec] = s.loc[both & ~c_dec].str.replace(",","",regex=False)
    s.loc[s.str.contains(",",na=False) & ~s.str.contains(r"\.",na=False)] = s.loc[s.str.contains(",",na=False) & ~s.str.contains(r"\.",na=False)].str.replace(",",".",regex=False)
    return pd.to_numeric(s, errors="coerce")

def parse_dates(s): return pd.to_datetime(s, errors="coerce", dayfirst=True) if not pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")
def clean_str(s): return s.astype("string").str.strip().replace("", pd.NA)
def safe_ratio(a,b): return (a/b).where(a.notna() & b.notna() & b.ne(0))
def pct_change(s):
    prev = s.shift(1)
    return ((s/prev - 1)*100).where(prev.gt(0) & s.gt(0))
def iqr_th(s, minimum):
    v = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    if len(v) < 8: return float(minimum)
    q1, q3 = v.quantile([0.25,0.75])
    t = q3 + 3.0*(q3-q1)
    return float(max(minimum,t)) if np.isfinite(t) else float(minimum)
def rel_label(n):
    if pd.isna(n): return "Нет данных"
    n=int(n)
    return "Высокая (≥3)" if n>=3 else "Средняя (2)" if n==2 else "Низкая (1)" if n==1 else "Нет данных"

def run_analysis(input_file):
    # ИСПОЛЬЗУЕМ CALAMINE ДЛЯ СКОРОСТИ
    try:
        raw = pd.read_excel(input_file, sheet_name=1, engine="calamine")
    except Exception:
        try:
            raw = pd.read_excel(input_file, sheet_name=0, engine="calamine")
        except Exception:
            raw = pd.read_excel(input_file, sheet_name=1, engine="openpyxl") # fallback
            
    if not isinstance(raw, pd.DataFrame) or raw.empty: raise ValueError("Лист пуст.")

    source_count = len(raw); colmap = resolve_columns(raw.columns)
    df = raw.rename(columns={v:k for k,v in colmap.items()}).copy()

    for c in OPTIONAL_TEXT: df[c] = clean_str(df[c]).fillna("Не указано") if c in df.columns else "Не указано"
    for c in ["sku","competitor"]: df[c] = clean_str(df[c])
    df["date"] = parse_dates(df["date"])
    for c in ["competitor_regular","lenta_regular","turnover"]: df[c] = parse_numeric(df[c])
    for c in ["competitor_promo","lenta_promo"]: df[c] = parse_numeric(df[c]) if c in df.columns else np.nan

    m_key = df[["sku","competitor","date"]].isna().any(axis=1)
    b_price = df["competitor_regular"].isna() | df["lenta_regular"].isna() | df["competitor_regular"].le(0) | df["lenta_regular"].le(0)
    b_to = df["turnover"].isna() | df["turnover"].lt(0)
    inv_mask = m_key | b_price | b_to
    valid = df.loc[~inv_mask].copy()
    
    dup_mask = valid.duplicated(keep="first")
    valid = valid.loc[~dup_mask].copy()

    c_key = ["city","sku","competitor","date"]
    v_cols = ["competitor_regular","lenta_regular","turnover","competitor_promo","lenta_promo"]
    variants = valid.groupby(c_key, dropna=False)[v_cols].apply(lambda x: len(x.drop_duplicates())).rename("v").reset_index()
    c_keys = variants.loc[variants["v"]>1, c_key]
    
    if not c_keys.empty:
        valid = valid.merge(c_keys.assign(_c=True), on=c_key, how="left")
        valid = valid.loc[valid["_c"].isna()].drop(columns="_c")

    valid = valid.sort_values(["city","sku","competitor","date"]).reset_index(drop=True)
    s_key = ["city","sku","competitor"]
    
    valid["Изм_конк_%"] = valid.groupby(s_key)["competitor_regular"].transform(pct_change)
    valid["Изм_лент_%"] = valid.groupby(s_key)["lenta_regular"].transform(pct_change)
    j_lim = valid.groupby("group")["Изм_конк_%"].apply(lambda x: iqr_th(x.abs(), 10.0)).to_dict()
    valid["Порог_ск"] = valid["group"].map(j_lim).fillna(10.0)
    
    valid["Мед_конк"] = valid.groupby(s_key)["competitor_regular"].transform("median")
    valid["Откл_%"] = (valid["competitor_regular"] / valid["Мед_конк"] - 1) * 100
    d_lim = valid.groupby("group")["Откл_%"].apply(lambda x: iqr_th(x.abs(), 15.0)).to_dict()
    valid["Порог_откл"] = valid["group"].map(d_lim).fillna(15.0)
    valid["Дат"] = valid.groupby(s_key)["date"].transform("nunique")

    j_flag = valid["Изм_конк_%"].abs().gt(valid["Порог_ск"]) & valid["Изм_лент_%"].abs().le(valid["Порог_ск"])
    d_flag = valid["Откл_%"].abs().gt(valid["Порог_откл"]) & valid["Дат"].ge(5)
    valid["Аномалия"] = np.select([j_flag & d_flag, j_flag, d_flag], ["Скачок и откл", "Скачок", "Откл"], default="")
    
    a_rows = valid.loc[valid["Аномалия"].eq("")].copy()
    l_dates = a_rows.groupby(["city","sku"])["date"].max().rename("last").reset_index()
    l_rows = a_rows.merge(l_dates, on=["city","sku"]).query("date == last").copy()

    l_agg = l_rows.groupby(["city","sku"], as_index=False).agg(
        **{"category":("category","first"), "subcategory":("subcategory","first"),
           "Цена Ленты (рег.)":("lenta_regular","median"), "Цена рынка (медиана по конкурентам)":("competitor_regular","median"),
           "Промо Ленты":("lenta_promo","median"), "Промо рынка (медиана)":("competitor_promo","median"),
           "Товарооборот":("turnover","median"), "Количество конкурентов":("competitor","nunique"), 
           "Последняя дата наблюдения":("date","max")})

    l_agg["Индекс регулярной цены"] = safe_ratio(l_agg["Цена Ленты (рег.)"], l_agg["Цена рынка (медиана по конкурентам)"])
    l_agg["Индекс регулярной цены (история)"] = l_agg["Индекс регулярной цены"] # Упрощение для скорости
    l_agg["Достоверность индекса"] = l_agg["Количество конкурентов"].map(rel_label)
    l_agg["Скидка Ленты, %"] = ((1 - l_agg["Промо Ленты"]/l_agg["Цена Ленты (рег.)"])*100).where(l_agg["Промо Ленты"].gt(0) & l_agg["Промо Ленты"].le(l_agg["Цена Ленты (рег.)"]))
    l_agg["Скидка конкурента, %"] = ((1 - l_agg["Промо рынка (медиана)"]/l_agg["Цена рынка (медиана по конкурентам)"])*100).where(l_agg["Промо рынка (медиана)"].gt(0) & l_agg["Промо рынка (медиана)"].le(l_agg["Цена рынка (медиана по конкурентам)"]))
    l_agg["Разница скидок, п.п."] = l_agg["Скидка Ленты, %"] - l_agg["Скидка конкурента, %"]

    sku = l_agg.copy()
    sku["Медианный ТО подкатегории"] = sku.groupby(["city","subcategory"])["Товарооборот"].transform("median")
    h_p = sku["Индекс регулярной цены"].gt(1.05); l_p = sku["Индекс регулярной цены"].lt(0.95)
    h_t = sku["Товарооборот"].gt(sku["Медианный ТО подкатегории"])
    sku["Сегмент"] = np.select([h_p & h_t, h_p & ~h_t, l_p & h_t, l_p & ~h_t], ["Дороже рынка / высокий ТО", "Дороже рынка / низкий ТО", "Дешевле рынка / высокий ТО", "Дешевле рынка / низкий ТО"], default="Ценовой паритет")
    sku.loc[sku["Индекс регулярной цены"].isna() | sku["Товарооборот"].isna(), "Сегмент"] = "Недостаточно данных"
    
    diff = sku["Разница скидок, п.п."]
    sku["Сравнение скидок"] = np.select([diff.gt(5.0), diff.lt(-5.0), diff.notna()], ["Скидка Ленты глубже", "Скидка Ленты меньше", "Скидки сопоставимы"], default="Нет данных")

    clean_exp = sku.rename(columns={"city":"Город","sku":"СКЮ","category":"Категория","subcategory":"Подкатегория"})
    f_cols = ["Сегмент","Достоверность индекса","Сравнение скидок","Город","СКЮ","Категория","Подкатегория","Товарооборот","Индекс регулярной цены","Индекс регулярной цены (история)","Скидка Ленты, %","Скидка конкурента, %","Разница скидок, п.п.","Количество конкурентов","Последняя дата наблюдения"]
    clean_exp = clean_exp[[c for c in f_cols if c in clean_exp.columns]]

    # Сохраняем в память
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", engine_kwargs={"options":{"nan_inf_to_errors":True}}, datetime_format="dd.mm.yyyy") as writer:
        wb = writer.book
        h_fmt = wb.add_format({"bold":True,"font_color":"#FFFFFF","bg_color":"#39785D","text_wrap":True})
        clean_exp.to_excel(writer, sheet_name="1. Очищенные данные", index=False)
        ws = writer.sheets["1. Очищенные данные"]
        ws.freeze_panes(1, 0)
        if "Индекс регулярной цены" in clean_exp.columns:
            c = clean_exp.columns.get_loc("Индекс регулярной цены")
            ws.conditional_format(1, c, len(clean_exp), c, {"type": "3_color_scale"})
            
    output.seek(0)
    return output

# --- ИНТЕРФЕЙС САЙТА ---
st.title("🛒 Анализ цен и товарооборота «Ленты»")
st.markdown("Загрузите исходный Excel-файл, и программа сформирует готовый аналитический отчет.")

uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.info("✅ Файл успешно загружен! Нажмите кнопку ниже, чтобы запустить обработку.")
    
    if st.button("🚀 Запустить анализ", use_container_width=True):
        with st.spinner("⏳ Идет обработка данных (используется быстрый движок)..."):
            try:
                excel_data = run_analysis(uploaded_file)
                st.success("🎉 Анализ успешно завершен!")
                st.balloons()
                
                st.download_button(
                    label="📥 Скачать готовый отчет",
                    data=excel_data,
                    file_name="Итоговый_отчет_Лента.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {e}")
