import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os

st.set_page_config(page_title="Анализ цен Ленты", page_icon="🛒", layout="centered")

# --- КОНСТАНТЫ И ЛОГИКА (ОРИГИНАЛЬНЫЙ КОД) ---
IQR_MULTIPLIER = 3.0
MIN_IQR_SAMPLE = 8
MIN_JUMP_THRESHOLD_PCT = 10.0
MIN_DEVIATION_THRESHOLD_PCT = 15.0
MIN_HISTORY_DATES = 5
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

def display_columns(df): return df.rename(columns=DISPLAY).copy()
def drop_empty_columns(df): return df.loc[:, df.notna().any()] if not df.empty else df

def write_large_df(writer, df, base_name, header_format, max_width=40):
    df = df.replace([np.inf,-np.inf], np.nan)
    chunks = [df] if len(df)==0 else [df.iloc[i:i+EXCEL_DATA_ROWS_PER_SHEET] for i in range(0, len(df), EXCEL_DATA_ROWS_PER_SHEET)]
    for part, chunk in enumerate(chunks,1):
        sheet = (base_name[:31-len(f"_{part}")] + (f"_{part}" if len(chunks)>1 else ""))
        chunk.to_excel(writer, sheet_name=sheet, index=False)
        ws = writer.sheets[sheet]; ws.freeze_panes(1,0); ws.set_default_row(18)
        if len(chunk.columns):
            ws.autofilter(0,0,len(chunk),len(chunk.columns)-1)
            for j, col in enumerate(chunk.columns):
                ws.write(0,j,str(col),header_format)
                sample = chunk[col].head(250).astype("string").fillna("")
                w = min(max(len(str(col))+2, max([len(x) for x in sample.tolist()] or [10])+2), max_width)
                name = str(col).lower(); fmt = None
                if "дата" in name or "обновление" in name: fmt = writer.book.add_format({"num_format":"dd.mm.yyyy"})
                elif "%" in name or "индекс" in name: fmt = writer.book.add_format({"num_format":"0.00"})
                elif "цена" in name or "товарооборот" in name or name=="то": fmt = writer.book.add_format({"num_format":"#,##0.00"})
                ws.set_column(j,j,w,fmt)

def run_analysis(input_path):
    try:
        raw = pd.read_excel(input_path, sheet_name=1, engine="calamine")
    except Exception:
        try: raw = pd.read_excel(input_path, sheet_name=0, engine="calamine")
        except Exception: raw = pd.read_excel(input_path, sheet_name=1, engine="openpyxl")

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
    inv = df.loc[inv_mask].copy()
    
    def _reason(i):
        parts = []
        if m_key.loc[i]: parts.append("Нет SKU/конкурента/даты")
        if b_price.loc[i]: parts.append("Некорректная цена")
        if b_to.loc[i]: parts.append("Некорректный ТО")
        return "; ".join(parts)
        
    inv["Причина исключения"] = [_reason(i) for i in inv.index]
    valid = df.loc[~inv_mask].copy()
    
    dup_mask = valid.duplicated(keep="first")
    dups = valid.loc[dup_mask].copy(); dups["Причина исключения"] = "Дубликат"
    valid = valid.loc[~dup_mask].copy()

    c_key = ["city","sku","competitor","date"]
    v_cols = ["competitor_regular","lenta_regular","turnover","competitor_promo","lenta_promo"]
    variants = valid.groupby(c_key, dropna=False)[v_cols].apply(lambda x: len(x.drop_duplicates())).rename("v").reset_index()
    c_keys = variants.loc[variants["v"]>1, c_key]
    
    if not c_keys.empty:
        c_rows = valid.merge(c_keys, on=c_key, how="inner"); c_rows["Причина исключения"] = "Конфликт"
        valid = valid.merge(c_keys.assign(_c=True), on=c_key, how="left")
        valid = valid.loc[valid["_c"].isna()].drop(columns="_c")
    else:
        c_rows = valid.iloc[0:0].copy(); c_rows["Причина исключения"] = pd.Series(dtype="object")

    valid = valid.sort_values(["city","sku","competitor","date"]).reset_index(drop=True)
    s_key = ["city","sku","competitor"]
    
    valid["Изм_конк_%"] = valid.groupby(s_key)["competitor_regular"].transform(pct_change)
    valid["Изм_лент_%"] = valid.groupby(s_key)["lenta_regular"].transform(pct_change)
    
    j_lim = valid.groupby("group")["Изм_конк_%"].apply(lambda x: iqr_th(x.abs(), MIN_JUMP_THRESHOLD_PCT)).to_dict()
    valid["Порог_ск"] = valid["group"].map(j_lim).fillna(MIN_JUMP_THRESHOLD_PCT)
    
    valid["Мед_конк"] = valid.groupby(s_key)["competitor_regular"].transform("median")
    valid["Откл_%"] = (valid["competitor_regular"] / valid["Мед_конк"] - 1) * 100
    
    d_lim = valid.groupby("group")["Откл_%"].apply(lambda x: iqr_th(x.abs(), MIN_DEVIATION_THRESHOLD_PCT)).to_dict()
    valid["Порог_откл"] = valid["group"].map(d_lim).fillna(MIN_DEVIATION_THRESHOLD_PCT)
    valid["Дат"] = valid.groupby(s_key)["date"].transform("nunique")

    j_flag = valid["Изм_конк_%"].abs().gt(valid["Порог_ск"]) & valid["Изм_лент_%"].abs().le(valid["Порог_ск"])
    d_flag = valid["Откл_%"].abs().gt(valid["Порог_откл"]) & valid["Дат"].ge(MIN_HISTORY_DATES)
    valid["Ценовая аномалия"] = np.select([j_flag & d_flag, j_flag, d_flag], ["Скачок и отклонение", "Резкий скачок", "Отклонение"], default="")
    
    anom = valid.loc[valid["Ценовая аномалия"].ne("")].copy(); anom["Причина исключения"] = "Аномалия: " + anom["Ценовая аномалия"]
    a_rows = valid.loc[valid["Ценовая аномалия"].eq("")].copy()

    l_dates = a_rows.groupby(["city","sku"])["date"].max().rename("last").reset_index()
    l_rows = a_rows.merge(l_dates, on=["city","sku"]).query("date == last").copy()

    l_agg = l_rows.groupby(["city","sku"], as_index=False).agg(
        **{"category":("category","first"), "subcategory":("subcategory","first"), "group":("group","first"),
           "Цена Ленты (рег.)":("lenta_regular","median"), "Цена рынка (медиана по конкурентам)":("competitor_regular","median"),
           "Промо Ленты":("lenta_promo","median"), "Промо рынка (медиана)":("competitor_promo","median"),
           "Товарооборот":("turnover","median"), "Количество конкурентов":("competitor","nunique"), 
           "Последняя дата наблюдения":("date","max")})

    l_agg["Индекс регулярной цены"] = safe_ratio(l_agg["Цена Ленты (рег.)"], l_agg["Цена рынка (медиана по конкурентам)"])
    # УПРОЩЕНИЕ ДЛЯ СКОРОСТИ САЙТА (истории все равно нет)
    l_agg["Индекс регулярной цены (история)"] = l_agg["Индекс регулярной цены"]
    
    l_agg["Достоверность индекса"] = l_agg["Количество конкурентов"].map(rel_label)
    l_agg["Скидка Ленты, %"] = ((1 - l_agg["Промо Ленты"]/l_agg["Цена Ленты (рег.)"])*100).where(l_agg["Промо Ленты"].gt(0) & l_agg["Промо Ленты"].le(l_agg["Цена Ленты (рег.)"]))
    l_agg["Скидка конкурента, %"] = ((1 - l_agg["Промо рынка (медиана)"]/l_agg["Цена рынка (медиана по конкурентам)"])*100).where(l_agg["Промо рынка (медиана)"].gt(0) & l_agg["Промо рынка (медиана)"].le(l_agg["Цена рынка (медиана по конкурентам)"]))
    l_agg["Разница скидок, п.п."] = l_agg["Скидка Ленты, %"] - l_agg["Скидка конкурента, %"]

    sku = l_agg.copy()
    sku["Медианный ТО подкатегории"] = sku.groupby(["city","subcategory"])["Товарооборот"].transform("median")
    h_p = sku["Индекс регулярной цены"].gt(PRICE_INDEX_HIGH); l_p = sku["Индекс регулярной цены"].lt(PRICE_INDEX_LOW)
    h_t = sku["Товарооборот"].gt(sku["Медианный ТО подкатегории"])
    sku["Сегмент"] = np.select([h_p & h_t, h_p & ~h_t, l_p & h_t, l_p & ~h_t], ["Дороже рынка / высокий ТО", "Дороже рынка / низкий ТО", "Дешевле рынка / высокий ТО", "Дешевле рынка / низкий ТО"], default="Ценовой паритет")
    sku.loc[sku["Индекс регулярной цены"].isna() | sku["Товарооборот"].isna(), "Сегмент"] = "Недостаточно данных"
    
    diff = sku["Разница скидок, п.п."]
    sku["Сравнение скидок"] = np.select([diff.gt(DISCOUNT_DIFF_PP), diff.lt(-DISCOUNT_DIFF_PP), diff.notna()], ["Скидка Ленты глубже", "Скидка Ленты меньше", "Скидки сопоставимы"], default="Нет данных")

    cat = sku.groupby(["city","category"], as_index=False, dropna=False).agg(**{"Товарооборот":("Товарооборот","sum"), "Количество SKU":("sku","nunique"), "Медианный индекс цены":("Индекс регулярной цены","median"), "Дороже рынка / низкий ТО":("Сегмент", lambda x: int((x=="Дороже рынка / низкий ТО").sum())), "Дешевле рынка / высокий ТО":("Сегмент", lambda x: int((x=="Дешевле рынка / высокий ТО").sum()))})
    cat["Доля ТО в городе, %"] = safe_ratio(cat["Товарооборот"], cat.groupby("city")["Товарооборот"].transform("sum"))*100

    r_cat = cat.loc[cat["Количество SKU"]>=MIN_SKU_FOR_CATEGORY_RANKING].sort_values(["Дороже рынка / низкий ТО","Товарооборот"], ascending=[False,False]).head(10)
    o_cat = cat.loc[cat["Количество SKU"]>=MIN_SKU_FOR_CATEGORY_RANKING].sort_values(["Дешевле рынка / высокий ТО","Товарооборот"], ascending=[False,False]).head(10)

    s_cols = ["city","sku","category","subcategory","Товарооборот","Индекс регулярной цены","Сегмент","Сравнение скидок","Количество конкурентов"]
    r_sku = sku.loc[sku["Сегмент"]=="Дороже рынка / низкий ТО"].sort_values(["Количество конкурентов","Товарооборот"], ascending=[False,False]).head(15)[s_cols]
    o_sku = sku.loc[sku["Сегмент"]=="Дешевле рынка / высокий ТО"].sort_values(["Количество конкурентов","Товарооборот"], ascending=[False,False]).head(15)[s_cols]

    r_cnt = int((sku["Сегмент"]=="Дороже рынка / низкий ТО").sum())
    o_cnt = int((sku["Сегмент"]=="Дешевле рынка / высокий ТО").sum())

    excl_parts = [x for x in [inv, dups, c_rows, anom] if not x.empty]
    excl = pd.concat(excl_parts, ignore_index=True, sort=False) if excl_parts else pd.DataFrame()
    if not excl.empty:
        b_cols = ["city","sku","category","subcategory","group","date","competitor","competitor_regular","lenta_regular","turnover","competitor_promo","lenta_promo","Причина исключения"]
        excl_exp = display_columns(excl[[c for c in b_cols if c in excl.columns]])
        excl_exp = drop_empty_columns(excl_exp)
    else:
        excl_exp = pd.DataFrame(columns=["Город","СКЮ","Категория","Причина исключения"])

    clean_exp = sku.rename(columns={"city":"Город","sku":"СКЮ","category":"Категория","subcategory":"Подкатегория","group":"Группа"})
    f_cols = ["Сегмент","Достоверность индекса","Сравнение скидок","Город","СКЮ","Категория","Подкатегория","Товарооборот","Индекс регулярной цены","Индекс регулярной цены (история)","Скидка Ленты, %","Скидка конкурента, %","Разница скидок, п.п.","Количество конкурентов","Последняя дата наблюдения"]
    clean_exp = clean_exp[[c for c in f_cols if c in clean_exp.columns]]

    stats = pd.DataFrame([("Всего строк в файле:", int(source_count)), ("Прошли проверку:", int(len(a_rows))), (f"Исключено: {len(inv)+len(dups)+len(c_rows)+len(anom)}", "")], columns=["Показатель","Значение"])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", engine_kwargs={"options":{"nan_inf_to_errors":True}}, datetime_format="dd.mm.yyyy") as writer:
        wb = writer.book
        title_fmt = wb.add_format({"bold":True,"font_size":15,"font_color":"#FFFFFF","bg_color":"#245B45","valign":"vcenter"})
        section_fmt = wb.add_format({"bold":True,"font_size":12,"font_color":"#FFFFFF","bg_color":"#39785D","valign":"vcenter"})
        header_fmt = wb.add_format({"bold":True,"font_color":"#FFFFFF","bg_color":"#39785D","text_wrap":True,"valign":"vcenter"})
        wrap_fmt = wb.add_format({"text_wrap":True,"valign":"top"})
        kpi_label = wb.add_format({"bold":True,"font_size":11,"valign":"vcenter"})
        num_int = wb.add_format({"num_format":"#,##0","valign":"vcenter"})
        
        ws = wb.add_worksheet("1. Сводка"); writer.sheets["1. Сводка"] = ws
        ws.set_column("A:A", 40); ws.set_column("B:B", 70, wrap_fmt)
        ws.merge_range(0,0,0,1,"Анализ цен и товарооборота «Ленты»", title_fmt); ws.set_row(0, 28)
        ws.merge_range(2,0,2,1,"Результат", section_fmt)
        ws.write(3,0,"Требуют внимания (дорого и не берут):", kpi_label); ws.write_number(3,1,r_cnt,num_int)
        ws.write(4,0,"Есть потенциал (дешево и берут):", kpi_label); ws.write_number(4,1,o_cnt,num_int)
        
        ws.merge_range(6,0,6,1,"Статистика обработки", section_fmt)
        ws.write(7,0,"Показатель", header_fmt); ws.write(7,1,"Значение", header_fmt)
        for i, r in stats.iterrows():
            ws.write(8+i, 0, r["Показатель"], wrap_fmt)
            if r["Значение"] != "": ws.write_number(8+i, 1, int(r["Значение"]), num_int)

        pws = wb.add_worksheet("2. Приоритеты"); writer.sheets["2. Приоритеты"] = pws
        def _write_block(row, title, df):
            pws.merge_range(row,0,row,max(1,len(df.columns)),title, section_fmt); row+=1
            df.to_excel(writer, sheet_name="2. Приоритеты", index=False, startrow=row)
            for j,c in enumerate(df.columns): pws.write(row,j,str(c),header_fmt)
            return row + len(df) + 3
            
        row = 0
        row = _write_block(row, "КАТЕГОРИИ: Риск", r_cat.rename(columns={"city":"Город","category":"Категория"}))
        row = _write_block(row, "SKU: Риск", r_sku.rename(columns={"city":"Город","sku":"СКЮ","category":"Категория","subcategory":"Подкатегория"}))
        row = _write_block(row, "КАТЕГОРИИ: Потенциал", o_cat.rename(columns={"city":"Город","category":"Категория"}))
        _write_block(row, "SKU: Потенциал", o_sku.rename(columns={"city":"Город","sku":"СКЮ","category":"Категория","subcategory":"Подкатегория"}))

        write_large_df(writer, clean_exp, "3. Очищенные данные", header_fmt)
        write_large_df(writer, excl_exp, "4. Исключенные", header_fmt)
        
    output.seek(0)
    return output

# --- ИНТЕРФЕЙС САЙТА ---
st.title("🛒 Анализ цен и товарооборота «Ленты»")
st.markdown("Загрузите исходный Excel-файл, и программа сформирует готовый аналитический отчет.")

uploaded_file = st.file_uploader("Выберите Excel-файл (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.info("✅ Файл загружен! Нажмите кнопку ниже.")
    
    if st.button("🚀 Запустить анализ", use_container_width=True):
        with st.spinner("⏳ Идет обработка данных..."):
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
                st.error(f"❌ Ошибка: {e}")
