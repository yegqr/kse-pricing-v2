"""
KSE Pricing Optimizer — Streamlit UI  (light theme)
Потребує: kse_pricing.py у тій самій директорії

Запуск:  streamlit run kse_pricing_app.py
"""

import os, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from kse_pricing import (
    load_and_clean, fit_global_ols, bootstrap_beta1,
    compute_rho_cascade, optimize_program,
    UAH_PER_USD, CPI_TO_2026,
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

UNI_LABELS = {
    "НаУКМА": "Києво-Могилянська",
    "УКУ":    "католицький",
    "AUK":    "Американ",
    "IT STEP": "STEP",
}


def apply_filters(df, years_sel, included_uni_substrings):
    is_kse = df["університет_назва"].str.contains("Київська школа")
    if years_sel:
        df = df[df["рік"].isin(years_sel)].copy()
    if included_uni_substrings is not None:
        keep = pd.Series(False, index=df.index)
        for s in included_uni_substrings:
            keep |= df["університет_назва"].str.contains(s, case=False)
        df = df[is_kse | (~is_kse & keep)].copy()
    return df


def run_model(df, MC, kse_year, n_boot):
    kse_all  = df[df["університет_назва"].str.contains("Київська школа")].copy()
    peer_all = df[~df["університет_назва"].str.contains("Київська школа")].copy()

    model, beta1, spec_cols, bp_pval, max_vif = fit_global_ols(df)
    boot_betas = bootstrap_beta1(df, spec_cols, n_boot=n_boot)
    b10, b50, b90 = np.percentile(boot_betas, [10, 50, 90])

    rho_by_spec, overall_rho = compute_rho_cascade(kse_all)

    hist_fp = {}
    for prog, grp in kse_all.groupby("освітня_програма"):
        hist_fp[prog] = {int(r["рік"]): int(r["fullpay"]) for _, r in grp.iterrows()}

    kse_target = df[
        df["університет_назва"].str.contains("Київська школа") &
        (df["рік"] == kse_year)
    ].copy()

    if kse_target.empty:
        return None, None

    results = []
    for _, row in kse_target.iterrows():
        spec = row["spec_group"]
        rho  = rho_by_spec.get(spec, overall_rho)
        peer_spec = peer_all[peer_all["spec_group"] == spec]
        res = optimize_program(row, beta1, boot_betas, rho, MC,
                               len(peer_spec), peer_spec["університет_назва"].nunique())
        res["hist_fp"] = hist_fp.get(row["освітня_програма"], {})
        # Єдина метрика прибутку: vs факт (чесне порівняння, не через ρ_avg)
        res["profit_fact_M"] = round((res["p_cur_k"] * 1000 - MC) * res["fp_actual_2025"] / 1e6, 3)
        res["delta_profit_vs_fact"] = round(
            (res["profit_opt_M"] / res["profit_fact_M"] - 1) * 100, 1
        ) if res["profit_fact_M"] > 0 else float("nan")
        results.append(res)

    reg_info = {
        "beta1": beta1, "adj_r2": model.rsquared_adj,
        "p_beta1": model.pvalues["price_2026"], "bp_pval": bp_pval,
        "max_vif": max_vif, "N": len(df), "b1_ci": (b10, b50, b90),
    }
    return results, reg_info


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS  (light theme)
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="KSE Pricing", layout="wide", page_icon="📊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]        { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3                        { font-family: 'IBM Plex Mono', monospace !important; color: #111827 !important; }
.stApp                            { background: #f9fafb; color: #111827; }
section[data-testid="stSidebar"]  { background: #ffffff; border-right: 1px solid #e5e7eb; }

/* metric cards */
.mc { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.mc-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; }
.mc-val   { font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: #111827; font-weight: 600; margin: 4px 0 2px; }
.mc-sub   { font-size: 12px; color: #9ca3af; }

/* step blocks */
.step { background: #ffffff; border-left: 3px solid #2563eb; padding: 14px 18px; margin-bottom: 10px; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.step-num   { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #2563eb; letter-spacing: .1em; }
.step-title { font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: #111827; font-weight: 600; margin: 3px 0 6px; }
.step-desc  { font-size: 13px; color: #4b5563; line-height: 1.6; }

/* column explanation */
.ce { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; margin-bottom: 6px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.ce-name    { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #2563eb; font-weight: 600; }
.ce-title   { font-size: 13px; color: #111827; font-weight: 600; margin-top: 2px; }
.ce-desc    { font-size: 12px; color: #6b7280; margin-top: 3px; line-height: 1.5; }
.ce-formula { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #16a34a; margin-top: 5px; }

/* data section */
.data-block { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px 20px; margin-bottom: 10px; }
.data-title { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #1d4ed8; font-weight: 600; margin-bottom: 6px; }
.data-text  { font-size: 13px; color: #1e3a5f; line-height: 1.6; }

/* status badges */
.pub  { background: #dcfce7; color: #166534; padding: 2px 9px; border-radius: 4px; font-family: monospace; font-size: 12px; border: 1px solid #bbf7d0; }
.skip { background: #fef3c7; color: #92400e; padding: 2px 9px; border-radius: 4px; font-family: monospace; font-size: 12px; border: 1px solid #fde68a; }

/* warning box */
.warn { background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 10px 14px; font-size: 13px; color: #78350f; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📁 Дані")
    uploaded = st.file_uploader("CSV (vstup_*.csv)", type=["csv"], label_visibility="collapsed")

    # Дефолтний файл — якщо нічого не завантажено, шукаємо vstup_22-25.csv поруч
    DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "vstup_22-25.csv")
    if uploaded is None:
        if os.path.exists(DEFAULT_CSV):
            st.caption(f"📂 Використовується: vstup_22-25.csv")
        else:
            st.caption("⚠ vstup_22-25.csv не знайдено поруч зі скриптом")

    st.markdown("---")
    st.markdown("### ⚙️ Параметри моделі")
    mc_usd   = st.number_input("MC (маржинальні витрати), USD", 0, 20000, 2000, 100)
    mc_uah   = mc_usd * UAH_PER_USD
    st.caption(f"= {mc_uah:,.0f} грн  (× {UAH_PER_USD} UAH/$)")
    kse_year = st.selectbox("Рік KSE для оптимізації", [2025, 2024, 2023])
    n_boot   = st.slider("Bootstrap ітерацій", 200, 2000, 500, 100)

    st.markdown("---")
    st.markdown("### 🗓 Роки у вибірці")
    years_all = [2022, 2023, 2024, 2025]
    years_sel = [y for y in years_all if st.checkbox(str(y), value=True, key=f"y{y}")]

    st.markdown("---")
    st.markdown("### 🏛 Peer-університети")
    st.caption("Зніміть — виключити з регресії")
    inc_subs = [sub for lbl, sub in UNI_LABELS.items()
                if st.checkbox(lbl, value=True, key=f"u{lbl}")]

    run_btn = st.button("▶ Запустити", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("# KSE Pricing Optimizer")
st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:14px;'>"
            "OLS demand estimation · anchored profit-max · bootstrap CI</p>",
            unsafe_allow_html=True)

if not uploaded and not os.path.exists(DEFAULT_CSV):
    st.info("⬅ Завантажте CSV у sidebar або покладіть vstup_22-25.csv поруч зі скриптом")
    st.stop()

if not run_btn:
    st.markdown("<p style='color:#9ca3af'>Налаштуйте параметри і натисніть **▶ Запустити**</p>",
                unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

with st.spinner("Завантаження та очищення даних..."):
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
    else:
        tmp_path = DEFAULT_CSV
    df_full = load_and_clean(tmp_path)
    df = apply_filters(
        df_full,
        years_sel if set(years_sel) != set(years_all) else None,
        inc_subs  if len(inc_subs) < len(UNI_LABELS) else None,
    )

with st.spinner(f"Регресія + bootstrap ({n_boot} ітерацій)..."):
    results, reg = run_model(df, mc_uah, kse_year, n_boot)

if results is None:
    st.error(f"Немає KSE-програм для {kse_year} у відфільтрованих даних.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# БЛОК 1 — ТАБЛИЦЯ РЕЗУЛЬТАТІВ
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 1 · Результати")

# Попередження про суперечність якщо оптимум < факту загалом
fact_total = sum(r["profit_fact_M"]  for r in results)
opt_total  = sum(r["profit_opt_M"] if r["status"] == "PUBLISH" else r["profit_fact_M"] for r in results)
if opt_total < fact_total:
    st.markdown(f"""<div class="warn">
    ⚠️ <b>Сумарний оптимум ({opt_total:.2f}M) менший за факт {kse_year} ({fact_total:.2f}M).</b>
    Це означає одне з двох: (а) поточні ціни вже близькі до оптимуму і модель рекомендує мінімальні зміни,
    або (б) ρ_all_years занижує baseline, і реальний виграш буде більшим. Interpret with care.
    </div>""", unsafe_allow_html=True)

rows = [{
    "Програма":      r["Програма"],
    "p_факт_k":      r["p_fact_k"],
    "p_cur_k":       r["p_cur_k"],
    "p_opt_k":       r["p_opt_k"],
    "Δp%":           r["delta_p_pct"],
    "fp_факт25":     r["fp_actual_2025"],
    "fp_прогноз":    r["fp_opt"],
    "CI_p10":        r["ci10_k"],
    "CI_p90":        r["ci90_k"],
    "ε_попит":       r["eps"],
    "Δprofit%_факт": r["delta_profit_vs_fact"],
    "profit_факт_M": r["profit_fact_M"],
    "profit_opt_M":  r["profit_opt_M"],
    "peers":         f"{r['peer_rows']}/{r['peer_unis']}",
    "CI_width%":     r["ci_width_pct"],
    "status":        r["status"],
} for r in results]

df_res = pd.DataFrame(rows)

def _style(df):
    def col_status(v):
        return ("background:#dcfce7;color:#166534" if v == "PUBLISH"
                else "background:#fef3c7;color:#92400e")
    def col_num(v):
        if pd.isna(v): return ""
        return "color:#16a34a;font-weight:600" if v > 0 else "color:#dc2626;font-weight:600"
    return (df.style
            .applymap(col_status, subset=["status"])
            .applymap(col_num, subset=["Δp%", "Δprofit%_факт"])
            .format({
                "p_факт_k": "{:.1f}k", "p_cur_k": "{:.1f}k", "p_opt_k": "{:.1f}k",
                "CI_p10": "{:.1f}k",   "CI_p90": "{:.1f}k",
                "Δp%": "{:+.1f}%",     "Δprofit%_факт": "{:+.1f}%",
                "fp_прогноз": "{:.1f}",
                "ε_попит": "{:.2f}",   "CI_width%": "{:.1f}%",
                "profit_факт_M": "{:.3f}M", "profit_opt_M": "{:.3f}M",
            }, na_rep="—"))

st.dataframe(_style(df_res), use_container_width=True, height=370)

s1, s2, s3 = st.columns(3)
base_total = sum(r["profit_base_M"] for r in results)
delta_vf   = (opt_total / fact_total - 1) * 100 if fact_total else 0

for col, lbl, val, sub in [
    (s1, f"Факт {kse_year}",   f"{fact_total:.2f}M грн", "(p_cur − MC) × fp_факт"),
    (s2, "Модельний baseline",  f"{base_total:.2f}M грн", "(p_cur − MC) × apps × ρ_avg ← занижено"),
    (s3, "Оптимум (PUBLISH)",   f"{opt_total:.2f}M грн",  f"vs факт: {delta_vf:+.1f}%"),
]:
    col.markdown(f'<div class="mc"><div class="mc-label">{lbl}</div>'
                 f'<div class="mc-val">{val}</div>'
                 f'<div class="mc-sub">{sub}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# БЛОК 2 — ПОЯСНЕННЯ КОЛОНОК
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 2 · Що означає кожна колонка")

cpi_note = "  ·  ".join(f"{yr}→2026: ×{mult}" for yr, mult in sorted(CPI_TO_2026.items()))

COL_DEFS = [
    ("p_факт_k",      "Номінальна ціна, тис. грн",
     "Реальна ціна у прайсі KSE. Без жодних коригувань.",
     "вартість_мін_грн / 1 000"),
    ("p_cur_k",       "CPI-приведена ціна → 2026, тис. грн",
     f"p_факт × CPI-множник конкретного року. {cpi_note}.",
     "p_факт × CPI(year→2026)"),
    ("p_opt_k",       "Рекомендована ціна 2026, тис. грн",
     "Ціна що максимізує (p − MC) × fullpay(p). Вже у цінах 2026 — більше не індексувати.",
     "argmax_p [(p − MC) · apps(p) · ρ]"),
    ("Δp%",           "Зміна ціни відносно p_cur",
     "На скільки % оптимальна ціна відрізняється від CPI-приведеної поточної.",
     "(p_opt / p_cur − 1) × 100"),
    ("fp_факт25",     f"Фактичних платників у {kse_year}",
     f"Реальна кількість студентів що заплатили контракт у {kse_year}. Не входить в оптимізацію — контекст.",
     "зараховано_платно  (ЄДЕБО)"),
    ("fp_прогноз",    "Прогноз платників при p_opt",
     "При підвищенні ціни — менше заяв → менше платників. Завжди менше fp_факт25 при рості ціни.",
     "(apps_cur + β₁·(p_opt − p_cur)) × ρ"),
    ("CI_p10 / CI_p90", "Bootstrap довірчий інтервал для p_opt",
     f"1000 resample → 1000 оптимумів → percentile 10% та 90%. Поточний запуск: {n_boot} ітерацій.",
     "percentile(p*_boot, [10, 90])"),
    ("ε_попит",       "Цінова еластичність заяв",
     "На 1% підвищення ціни — на скільки % зменшуються заяви. |ε| < 1 = нееластичний (є простір збільшити ціну). |ε| > 2 = еластичний.",
     "β₁ × p_cur / apps_cur"),
    ("Δprofit%_факт", f"Приріст profit vs факту {kse_year} ← основна метрика",
     f"Скільки % ми виграємо відносно реального {kse_year}. Якщо від'ємне — модель каже 'залишити ціну як є, ми не впевнені в зміні'.",
     f"(profit_opt / profit_факт{kse_year} − 1) × 100"),
    ("peers",         "Peer-рядків / університетів у spec_group",
     "Кількість спостережень для ідентифікації нахилу β₁. Мало → нахил оцінений переважно на одному унів.",
     "count(peer rows з тим самим spec_group)"),
    ("CI_width%",     "Ширина CI відносно p_cur",
     "Якщо > 30% → SKIP. Відображає невизначеність β₁ у bootstrap.",
     "(CI_p90 − CI_p10) / p_cur × 100"),
    ("status",        "Статус рекомендації",
     "PUBLISH = надійна рекомендація. SKIP (edge hit) = оптимум на межі гриду (±5% від краю) — лінійна екстраполяція вийшла за розумні межі. SKIP (unstable CI) = CI_width > 30%.",
     "edge_hit (5% від краю гриду)  OR  CI_width > 30%"),
]

lc, rc = st.columns(2)
for i, (nm, ttl, desc, formula) in enumerate(COL_DEFS):
    t = lc if i % 2 == 0 else rc
    t.markdown(
        f'<div class="ce"><div class="ce-name">{nm}</div>'
        f'<div class="ce-title">{ttl}</div>'
        f'<div class="ce-desc">{desc}</div>'
        f'<div class="ce-formula">= {formula}</div></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# БЛОК 3 — ОПИС ДАНИХ
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 3 · Дані: що, звідки і чому")

peer_unis_list = sorted(df[~df["університет_назва"].str.contains("Київська школа")]["університет_назва"].unique())
years_in_model = sorted(df["рік"].unique())
n_kse = len(df[df["університет_назва"].str.contains("Київська школа")])
n_peer = len(df[~df["університет_назва"].str.contains("Київська школа")])

st.markdown(f"""
<div class="data-block">
  <div class="data-title">📦 Поточна вибірка</div>
  <div class="data-text">
    <b>{len(df)} рядків</b> (рік × університет × програма) · роки: <b>{years_in_model}</b><br>
    KSE: <b>{n_kse} рядків</b> · Peer: <b>{n_peer} рядків</b> з <b>{len(peer_unis_list)} університетів</b><br>
    Peer-сет: {', '.join(peer_unis_list)}
  </div>
</div>
""", unsafe_allow_html=True)

data_blocks = [
    ("Джерело даних", "ЄДЕБО (Єдина державна електронна база освіти) — офіційна база MОН України. Містить дані про зарахування абітурієнтів по кожній програмі, кількість заяв, мінімальні конкурсні бали та вартість навчання."),
    ("Чому саме ці університети", "НаУКМА, УКУ, AUK, IT STEP — найближчий преміум peer-сет для KSE: приватні або автономні, орієнтовані на якісну освіту, мають контрактний набір. Масові ЗВО (КНУ, КПІ) не включені — у них принципово інша структура попиту через держзамовлення."),
    ("Чому контрактники = full-payers", "Для peer-університетів використовуємо зараховано_контракт як найближчий аналог fullpay. Припущення: всі контрактники платять повну ціну програми без знижок. Це виправдано для преміум-сегменту — гранти та стипендії тут рідкість. Для KSE беремо зараховано_платно напряму."),
    ("Чому 'premium segment' важливий для еластичності", "Ціль — оцінити еластичність попиту саме серед аудиторії, яка готова і здатна платити. Включення масових ЗВО змішало б два різні ринки. У преміум-сегменті рішення про університет менш чутливе до ціни — абітурієнт вже відфільтрований за платоспроможністю. Це і пояснює нееластичні ε близько −1."),
    ("Фільтри вибірки", "форма_навчання = Денна + на_базі = Повна загальна середня освіта. Відсіває вечірню, заочну та вступ після коледжу — ці ринки мають іншу цінову динаміку. Залишаємо тільки бакалаврат очної форми як єдиний порівнюваний сегмент."),
    ("Обмеження даних", "207 рядків — невелика вибірка. Ціна ендогенна (університети ставлять ціну спостерігаючи попит → Cov(p, u) ≠ 0). β₁ — best linear predictor, не причинний ефект. Рекомендації читаються як 'локально розумно при поточному стані ринку', не як 'точний прогноз набору'."),
]

lc, rc = st.columns(2)
for i, (title, text) in enumerate(data_blocks):
    t = lc if i % 2 == 0 else rc
    t.markdown(f'<div class="data-block"><div class="data-title">{title}</div>'
               f'<div class="data-text">{text}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# БЛОК 4 — РЕГРЕСІЯ
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 4 · Регресія")
st.caption(f"N = {reg['N']} спостережень · {len(peer_unis_list)} peer-університети · роки: {years_in_model}")

b10, b50, b90 = reg["b1_ci"]
c1, c2, c3, c4 = st.columns(4)
for col, lbl, val, sub in [
    (c1, "β₁  (нахил попиту)",  f"{reg['beta1']*1000:.4f}", "заяв / 1 000 грн"),
    (c2, "Adj. R²",              f"{reg['adj_r2']:.3f}",     f"N = {reg['N']}"),
    (c3, "p-value β₁",           f"{reg['p_beta1']:.4f}",    "HC3 SE · " + ("✓ < 0.01" if reg["p_beta1"] < 0.01 else "⚠ > 0.01")),
    (c4, "Bootstrap CI β₁",      f"[{b10*1000:.3f}, {b90*1000:.3f}]", f"median = {b50*1000:.4f}"),
]:
    col.markdown(f'<div class="mc"><div class="mc-label">{lbl}</div>'
                 f'<div class="mc-val">{val}</div>'
                 f'<div class="mc-sub">{sub}</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# БЛОК 5 — АЛГОРИТМ + LATEX
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 5 · Алгоритм моделі — крок за кроком")

STEPS = [
    ("00", "Що ми оцінюємо і навіщо",
     "Мета: знайти ціну p* що максимізує gross profit KSE по повних платниках. "
     "Для цього треба знати нахил кривої попиту β₁ — скільки абітурієнтів ми втрачаємо на кожну 1 грн підвищення ціни. "
     "β₁ не спостерігається напряму: KSE має одну ціну в рік. Тому ми позичаємо цінову варіацію у peer-ринку преміум-освіти — "
     "НаУКМА, УКУ, AUK, IT STEP — де ціни в тому ж сегменті різняться від 40k до 330k грн. "
     "Припущення: після контролю на спеціальність і селективність, залишкова варіація ціни між цими університетами "
     "відображає різні позиції на тій самій кривій попиту преміум-аудиторії.",
     r"\text{Мета: } \max_p \; \pi(p) = (p - MC) \cdot \underbrace{fullpay(p)}_{\text{платники при ціні } p}"),
    ("01", "CPI-нормалізація цін",
     "Ціни з різних років приводимо у гривні 2026. Без цього 80k у 2022 і 80k у 2025 — ідентичні точки, "
     "але купівельна спроможність різна: 80k у 2022 — це ≈100k у 2026.",
     r"p_{2026}^{(i,t)} = \text{вартість\_мін}^{(i,t)} \cdot \prod_{s=t}^{2025}(1+\pi_s)"),
    ("02", "Визначення fullpay та spec_group",
     "KSE: fullpay = зараховано_платно (прямо з ЄДЕБО). "
     "Peer: fullpay = зараховано_контракт — контрактники в преміум-сегменті платять повну суму без знижок. "
     "Двоетапний маппінг: спочатку освітня_програма, потім спеціальність — щоб AUK 'Бізнес-адміністрування' "
     "потрапила в ту ж group що і KSE 'Бізнес-економіка'.",
     r"\text{spec\_group} = \text{SPEC\_MAP}[\text{освітня\_програма}] \;\text{ або }\; \text{SPEC\_MAP}[\text{спеціальність}]"),
    ("03", "OLS з spec FE та HC3 SE",
     "Оцінюємо β₁ по всьому peer-ринку одночасно. Spec fixed effects прибирають 'бренд спеціальності' "
     "як середній рівень попиту — без цього медицина б порівнювалась з фізикою. "
     "HC3 стандартні помилки коригують гетероскедастичність (дисперсія залишків різна між програмами).",
     r"\text{apps}_{it} = \beta_0 + \beta_1 p_{it} + \beta_2 \text{priority}_{it} + \beta_3 \text{score}_{it} + \sum_k \gamma_k \mathbf{1}[\text{spec}=k] + u_{it}"),
    ("04", "Локальна еластичність у точці KSE",
     "β₁ — абсолютний нахил (заяв/грн). Переводимо у відносну еластичність: на 1% ціни — скільки % заяв. "
     "Ключове слово ЛОКАЛЬНА: це дотична до кривої в поточній точці, не властивість всієї кривої.",
     r"\varepsilon = \beta_1 \cdot \frac{p_{cur}}{apps_{cur}}"),
    ("05", "Anchored demand curve",
     "Криву попиту не відновлюємо глобально — це дало б нереалістичні числа для малих програм. "
     "Беремо реальну точку KSE і рухаємось вздовж оціненого нахилу. "
     "Це переводить модель з 'глобальний ринковий прогноз' у 'локальний price gradient навколо KSE 2025'.",
     r"\text{apps}(p) = \max\!\bigl(apps_{cur} + \beta_1 (p - p_{cur}),\ 0\bigr)"),
    ("06", "Конверсія заяв → платники через ρ",
     "ρ = середня конверсія заяв у платників по всіх роках KSE для spec_group. "
     "Важливо: ρ_all_years < ρ_2025 бо 2025 рік KSE мав аномально високу конверсію. "
     "Це занижує модельний baseline, але робить його більш консервативним.",
     r"\rho = \frac{\sum_t \text{fullpay}_t}{\sum_t \text{apps}_t}, \qquad \text{fullpay}(p) = \text{apps}(p) \cdot \rho"),
    ("07", "Оптимізація profit (grid search + Lerner)",
     "Максимізуємо gross profit. Grid 3000 точок від 0.75×p_cur до 1.6×p_cur. "
     "Якщо p* потрапляє в межах 5% від краю гриду — статус SKIP (edge hit): "
     "лінійна екстраполяція на ±60% від поточної точки не є надійною.",
     r"\pi(p) = (p - MC) \cdot \text{fullpay}(p), \qquad p^*_{\text{Lerner}} = \frac{p_{cur}+MC}{2} + \frac{apps_{cur}}{2|\beta_1|}"),
    ("08", f"Bootstrap невизначеності ({n_boot} ітерацій)",
     f"1000 resample панелю → переоцінюємо β₁ → збираємо розподіл p*. "
     "CI_width > 30% від p_cur → SKIP: рекомендація нестабільна, не публікувати.",
     r"p^*_{(b)} = \operatorname{argmax}_p\,\pi(p;\hat\beta_1^{(b)}), \quad CI = \bigl[p^*_{10\%},\; p^*_{90\%}\bigr]"),
]

for num, title, desc, latex in STEPS:
    st.markdown(
        f'<div class="step"><div class="step-num">Крок {num}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-desc">{desc}</div></div>',
        unsafe_allow_html=True
    )
    st.latex(latex)


# ─────────────────────────────────────────────────────────────
# БЛОК 6 — ДЕТАЛІ ПО ПРОГРАМАХ
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 6 · Деталі по кожній програмі")

for r in results:
    tag = '<span class="pub">PUBLISH</span>' if r["status"] == "PUBLISH" \
          else f'<span class="skip">{r["status"]}</span>'
    label = f"{r['Програма']}  ·  {r['p_fact_k']}k → {r['p_opt_k']}k  ({r['delta_p_pct']:+.1f}%)"

    with st.expander(label, expanded=False):
        hist = r["hist_fp"]
        hist_parts = [f"{yr}: **{cnt}**" for yr, cnt in sorted(hist.items()) if yr != kse_year]
        hist_str = "  →  ".join(hist_parts) + f"  →  **[{kse_year} факт: {r['fp_actual_2025']}]**" \
                   if hist_parts else f"**[{kse_year} факт: {r['fp_actual_2025']}]**"
        eps_c = "нееластичний — є простір для підвищення" if abs(r["eps"]) < 1.5 else "помірно еластичний"
        ci_c  = "✓ стабільна" if r["ci_width_pct"] <= 30 else "⚠ нестабільна"

        st.markdown(f"""
**Статус:** {tag} &nbsp;|&nbsp; **Spec group:** `{r['spec_group']}` &nbsp;|&nbsp; **Peers:** {r['peer_rows']} рядків / {r['peer_unis']} унів

---
**Ціна**

| | тис. грн |
|---|---|
| Факт {kse_year} | **{r['p_fact_k']}k** |
| CPI → 2026 | **{r['p_cur_k']}k** |
| Оптимум p* | **{r['p_opt_k']}k** ({r['delta_p_pct']:+.1f}%) |
| Bootstrap CI | [{r['ci10_k']}k – {r['ci90_k']}k] — CI_width {r['ci_width_pct']:.1f}% ({ci_c}) |
| Lerner-check | {r['p_lerner_k']}k |

**Платники (fullpay)**

{hist_str} → [прогноз p*: **{r['fp_opt']}**]

**Прибуток**

| | M грн |
|---|---|
| Факт {kse_year} | {r['profit_fact_M']:.3f}M |
| При p* (оптимум) | **{r['profit_opt_M']:.3f}M** |
| Δ vs факт | **{r['delta_profit_vs_fact']:+.1f}%** |

**ε_попит = {r['eps']}** — {eps_c}  
**ρ = {r['rho_allyears']}** — конверсія заяв у платників (all-years KSE)
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<p style='color:#9ca3af;font-size:11px;text-align:center'>"
            "KSE Pricing Optimizer · OLS + spec FE + HC3 · anchored demand · bootstrap CI · Lerner profit-max"
            "</p>", unsafe_allow_html=True)