"""
KSE Bachelor Pricing Optimization — v2
Виправлення: канонічний CPI, all-years ρ, медіанна імпутація, skip LOW confidence.

Запуск:
    python3 kse_pricing.py --csv vstup_final-2.csv --mc-usd 2000

Залежності:
    pip install pandas numpy statsmodels scipy
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 0. КОНСТАНТИ
# ─────────────────────────────────────────────

# Канонічні CPI-множники до 2026 (з model_logic.py)
# Значення: скільки гривень 2026 відповідає 1 грн року t
CPI_TO_2026 = {
    2022: 1.2942,
    2023: 1.2314,
    2024: 1.0994,
    2025: 1.0180,
}

UAH_PER_USD = 43.51525  # NBU квітень 2026

# Маппінг → canonical spec_group.
# Застосовується ДВОЕТАПНО: спочатку освітня_програма, потім спеціальність.
#
# Логіка включення:
#   ПРЯМІ замінники  — та сама спеціальність в іншому університеті
#   НЕПРЯМІ замінники — суміжні поля, куди реально йдуть абітурієнти
#                        що розглядали KSE-програму
SPEC_MAP = {
    # ════════════════════════════════════════════════════════
    # KSE ПРОГРАМИ
    # ════════════════════════════════════════════════════════
    "Бізнес-економіка":                       "Економіка",
    "Економіка і великі дані":                "Економіка",
    "Програмна інженерія та бізнес-аналіз":   "Інженерія програмного забезпечення",
    "Програмна інженерія":                    "Інженерія програмного забезпечення",
    "Кібербезпека":                           "Кібербезпека та захист інформації",
    "Право":                                  "Право",
    "Прикладна математика":                   "Прикладна математика",
    "Фізична математика":                     "Прикладна математика",
    "Психологія":                             "Психологія",
    "Штучний інтелект":                       "Комп'ютерні науки",

    # ════════════════════════════════════════════════════════
    # ЕКОНОМІКА
    # Прямі: економіка, фінанси, менеджмент, маркетинг
    # Непрямі: МЕВ, політ.лідерство+економіка, Етика-Політика-Економіка (УКУ)
    # ════════════════════════════════════════════════════════
    "Економіка":                                                  "Економіка",
    "Економіка та міжнародні економічні відносини":               "Економіка",
    "Маркетинг":                                                  "Економіка",
    "Менеджмент":                                                 "Економіка",
    "Фінанси, банківська справа та страхування":                  "Економіка",
    "Фінанси, банківська справа, страхування та фондовий ринок":  "Економіка",
    "Міжнародні економічні відносини":                            "Економіка",
    # AUK
    "Бізнес-адміністрування":                                     "Економіка",
    "Ґлобал менеджмент":                                          "Економіка",
    "Бізнес-адміністрування; Ґлобал менеджмент":                  "Економіка",
    "Глобальні фінанси":                                          "Економіка",
    # Непрямі
    "Політичне лідерство та економічна дипломатія":               "Економіка",
    "Етика-Політика-Економіка":                                   "Економіка",  # УКУ — явно econ+politics

    # ════════════════════════════════════════════════════════
    # ІНЖЕНЕРІЯ ПРОГРАМНОГО ЗАБЕЗПЕЧЕННЯ
    # Прямі: ІПЗ, системний аналіз, автоматизація
    # Непрямі: робототехніка, бізнес-аналітика
    # ════════════════════════════════════════════════════════
    "Інженерія програмного забезпечення":                         "Інженерія програмного забезпечення",
    "Інженерія програмного забезпечення та штучний інтелект":     "Інженерія програмного забезпечення",
    "Проєктування цифрового досвіду та штучний інтелект":         "Інженерія програмного забезпечення",
    "Системний аналіз":                                           "Інженерія програмного забезпечення",
    "Аналітика даних":                                            "Інженерія програмного забезпечення",
    "Інформаційні технології та бізнес-аналітика":                "Інженерія програмного забезпечення",
    "Автоматизація, комп'ютерно-інтегровані технології та робототехніка":
                                                                  "Інженерія програмного забезпечення",
    "Робототехніка":                                              "Інженерія програмного забезпечення",

    # ════════════════════════════════════════════════════════
    # КОМПʼЮТЕРНІ НАУКИ  (Штучний інтелект KSE)
    # Прямі: комп'ютерні науки
    # Непрямі: data science / AI-треки, системний аналіз+дані
    # ════════════════════════════════════════════════════════
    "Комп'ютерні науки":                                          "Комп'ютерні науки",
    "Системний аналіз та наука про дані":                         "Комп'ютерні науки",
    "Аналітика даних та штучний інтелект":                        "Комп'ютерні науки",
    "Інформаційні технології та аналітика рішень":                "Комп'ютерні науки",

    # ════════════════════════════════════════════════════════
    # КІБЕРБЕЗПЕКА
    # Прямі: кібербезпека (тільки НаУКМА в датасеті)
    # Непрямі: немає достатньо близьких у цьому peer-сеті —
    #          тому залишаємо тільки пряме
    # ════════════════════════════════════════════════════════
    "Аналіз вразливостей інформаційних систем":                   "Кібербезпека та захист інформації",
    "Кібербезпека та захист інформації":                          "Кібербезпека та захист інформації",

    # ════════════════════════════════════════════════════════
    # ПРАВО
    # Прямі: право (НаУКМА + УКУ) — 8/2, достатньо
    # Публ.адмін прибрано: при додаванні Adj.R² падав,
    # попит на управлінські програми інший за структурою
    # ════════════════════════════════════════════════════════
    "Право":                                                      "Право",

    # ════════════════════════════════════════════════════════
    # ПРИКЛАДНА МАТЕМАТИКА  (+ Фізична математика KSE)
    # Прямі: прикладна математика (НаУКМА)
    # Непрямі: фізика (НаУКМА) — пряма аудиторія для Фізмату KSE
    # ════════════════════════════════════════════════════════
    "Прикладна математика":                                       "Прикладна математика",
    "Комп`ютерна фізика":                                         "Прикладна математика",
    "Комп'ютерна фізика":                                         "Прикладна математика",
    "Фізика":                                                     "Прикладна математика",
    "Фізика біологічних систем":                                  "Прикладна математика",
    "Фізика та астрономія":                                       "Прикладна математика",

    # ════════════════════════════════════════════════════════
    # ПСИХОЛОГІЯ
    # Прямі: психологія (НаУКМА + УКУ + AUK) — 9/3, достатньо
    # Соціологія / Соціальна робота прибрані: при додаванні
    # Adj.R² падав з 0.51 до 0.41 — структура попиту інша
    # ════════════════════════════════════════════════════════
    "Психологія":                                                 "Психологія",
}

# Поріг публікації рекомендації
MIN_PEER_ROWS   = 3   # peer-рядків у spec_group
MIN_PEER_UNIS   = 2   # різних peer-університетів
MAX_CI_WIDTH    = 0.30  # max (CI_p90 − CI_p10) / p_cur; ширше → UNSTABLE

GRID_N    = 3_000
N_BOOT    = 1_000
SEED      = 42


# ─────────────────────────────────────────────
# 1. ЗАВАНТАЖЕННЯ І ОЧИЩЕННЯ
# ─────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Фільтри з методології
    df = df[df["форма_навчання"] == "Денна"].copy()
    df = df[df["на_базі"].str.contains("Повна", na=False)].copy()

    # CPI-нормалізація (канонічні множники)
    df["price_2026"] = df.apply(
        lambda r: r["вартість_мін_грн"] * CPI_TO_2026.get(r["рік"], 1.0), axis=1
    )

    # Визначення fullpay
    def _fullpay(row):
        if "Київська школа" in row["університет_назва"]:
            v = row["зараховано_платно"]
            return v if pd.notna(v) else row["зараховано_контракт"]
        return row["зараховано_контракт"]

    df["fullpay"] = df.apply(_fullpay, axis=1)
    df["apps"]    = df["всього_заяв"]

    # Двоетапний маппінг:
    # 1) пробуємо освітня_програма (ловить KSE і прямі назви peer-програм)
    # 2) якщо не знайдено — пробуємо спеціальність (ловить AUK "Бізнес-адміністрування" etc.)
    # 3) якщо і спеціальність не в мапі — залишаємо raw спеціальність
    df["spec_group"] = (
        df["освітня_програма"].map(SPEC_MAP)
        .fillna(df["спеціальність"].map(SPEC_MAP))
        .fillna(df["спеціальність"])
    )

    # Медіанна імпутація — ВАЖЛИВО: global median, не per-spec.
    # середній_пріоритет має 85/207 пропусків (~41%). Імпутація per-spec_group
    # дала б одне значення на групу → лінійна залежність від spec FE → p(β₁)=1.
    # рейт_бал_контракт: 4 пропуски, глобальна медіана безпечна.
    for col in ["рейт_бал_контракт", "середній_пріоритет"]:
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

    # Виключити рядки без apps / fullpay / price
    df = df.dropna(subset=["apps", "fullpay", "price_2026"])
    df = df[df["apps"] > 0]

    return df


# ─────────────────────────────────────────────
# 2. ГЛОБАЛЬНА OLS З SPEC FE
# ─────────────────────────────────────────────

def fit_global_ols(df: pd.DataFrame):
    spec_dummies = pd.get_dummies(
        df["spec_group"], drop_first=True, prefix="s"
    ).astype(float)
    spec_cols = spec_dummies.columns.tolist()

    X = pd.concat(
        [df["price_2026"], df["середній_пріоритет"], df["рейт_бал_контракт"], spec_dummies],
        axis=1,
    )
    X = sm.add_constant(X)
    y = df["apps"]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Breusch–Pagan
    bp_lm, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)

    # VIF (тільки числові регресори без дамм і константи)
    vif_cols = ["price_2026", "середній_пріоритет", "рейт_бал_контракт"]
    vif_X = sm.add_constant(df[vif_cols]).values
    vif_vals = [variance_inflation_factor(vif_X, i) for i in range(1, len(vif_cols) + 1)]
    max_vif = max(vif_vals)

    return model, model.params["price_2026"], spec_cols, bp_pval, max_vif


# ─────────────────────────────────────────────
# 3. BOOTSTRAP β₁
# ─────────────────────────────────────────────

def bootstrap_beta1(df: pd.DataFrame, spec_cols: list, n_boot: int = N_BOOT) -> np.ndarray:
    np.random.seed(SEED)
    betas = []
    n = len(df)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        dfsb = df.iloc[idx].copy()
        sd = pd.get_dummies(dfsb["spec_group"], drop_first=True, prefix="s").astype(float)
        for c in spec_cols:
            if c not in sd.columns:
                sd[c] = 0.0
        Xb = pd.concat(
            [dfsb["price_2026"], dfsb["середній_пріоритет"], dfsb["рейт_бал_контракт"], sd[spec_cols]],
            axis=1,
        )
        Xb = sm.add_constant(Xb)
        try:
            b = sm.OLS(dfsb["apps"], Xb).fit().params.get("price_2026", np.nan)
            if pd.notna(b) and b < 0:
                betas.append(b)
        except Exception:
            pass

    return np.array(betas)


# ─────────────────────────────────────────────
# 4. ρ — КАСКАДНЕ ОБЧИСЛЕННЯ (all-years KSE)
# ─────────────────────────────────────────────

def compute_rho_cascade(kse_all: pd.DataFrame) -> dict:
    """
    Повертає словник spec_group → ρ_all_years.
    Каскад: program-level → spec-level → overall.
    """
    overall_rho = kse_all["fullpay"].sum() / kse_all["apps"].sum()

    rho_by_spec = {}
    for spec, grp in kse_all.groupby("spec_group"):
        if grp["apps"].sum() > 0:
            rho_by_spec[spec] = grp["fullpay"].sum() / grp["apps"].sum()
        else:
            rho_by_spec[spec] = overall_rho

    return rho_by_spec, overall_rho


# ─────────────────────────────────────────────
# 5. ОПТИМІЗАЦІЯ ПО ПРОГРАМІ
# ─────────────────────────────────────────────

def optimize_program(
    row: pd.Series,
    beta1: float,
    boot_betas: np.ndarray,
    rho: float,
    MC: float,
    peer_rows: int,
    peer_unis: int,
) -> dict:
    p_cur    = row["price_2026"]
    p_fact   = row["вартість_мін_грн"]          # номінальна ціна 2025 без CPI
    apps_cur = row["apps"]
    fp_actual_2025 = row["fullpay"]   # фактичний 2025, тільки як контекст

    # Profit-функції (model-consistent: однаковий ρ для baseline і оптимуму)
    grid = np.linspace(max(p_cur * 0.75, MC + 5_000), p_cur * 1.6, GRID_N)

    def _profit(p_arr):
        apps_ = np.maximum(apps_cur + beta1 * (p_arr - p_cur), 0)
        return (p_arr - MC) * apps_ * rho

    profit_grid = _profit(grid)
    idx_opt     = np.argmax(profit_grid)
    p_opt       = grid[idx_opt]
    profit_opt  = profit_grid[idx_opt]
    profit_base = _profit(np.array([p_cur]))[0]   # model-consistent baseline

    # Аналітична перевірка (Lerner)
    p_lerner = (p_cur + MC) / 2 + apps_cur / (2 * abs(beta1))

    # Bootstrap CI для p_opt
    p_opts_boot = []
    for b in boot_betas:
        pg = _profit.__func__ if False else None  # inline
        ag = np.maximum(apps_cur + b * (grid - p_cur), 0)
        profit_b = (grid - MC) * ag * rho
        p_opts_boot.append(grid[np.argmax(profit_b)])
    p_opts_boot = np.array(p_opts_boot)
    ci10, ci50, ci90 = np.percentile(p_opts_boot, [10, 50, 90])

    # Прапори
    # Edge hit: оптимум в межах 5% від краю гриду → лінійна екстраполяція вийшла за межі
    # Абсолютний поріг 2 000 грн не ловить при великих цінах (напр. 212k vs 216k = 4k > 2k)
    grid_range = grid[-1] - grid[0]
    edge_hit = (abs(p_opt - grid[0]) < 0.05 * grid_range) or (abs(p_opt - grid[-1]) < 0.05 * grid_range)
    ci_width_rel = (ci90 - ci10) / p_cur
    unstable = ci_width_rel > MAX_CI_WIDTH

    if edge_hit:
        publish = "SKIP (edge hit)"
    elif unstable:
        publish = "SKIP (unstable CI)"
    else:
        publish = "PUBLISH"

    # Еластичність попиту (заяв) — знаменник apps_cur, бо β₁ оцінений саме на заявах.
    # ε_контракт = ε_заяв при константному ρ (ρ скорочується з чисельника і знаменника).
    eps = beta1 * p_cur / apps_cur

    return {
        "Програма":        row["освітня_програма"],
        "spec_group":      row["spec_group"],
        "p_cur_k":         round(p_cur / 1_000, 1),
        "p_fact_k":        round(p_fact / 1_000, 1),
        "p_opt_k":         round(p_opt / 1_000, 1),
        "p_lerner_k":      round(p_lerner / 1_000, 1),
        "ci10_k":          round(ci10 / 1_000, 1),
        "ci90_k":          round(ci90 / 1_000, 1),
        "delta_p_pct":     round((p_opt / p_cur - 1) * 100, 1),
        "eps":             round(eps, 2),
        "rho_allyears":    round(rho, 3),
        "apps_cur":        int(apps_cur),
        "apps_opt":        int(round(float(np.maximum(apps_cur + beta1 * (p_opt - p_cur), 0)))),
        "fp_actual_2025":  int(fp_actual_2025),
        "fp_model_base":   round(apps_cur * rho, 1),
        "fp_opt":          round(float(np.maximum(apps_cur + beta1 * (p_opt - p_cur), 0) * rho), 1),
        "profit_base_M":   round(profit_base / 1e6, 3),
        "profit_opt_M":    round(profit_opt / 1e6, 3),
        "delta_profit_pct": round((profit_opt / profit_base - 1) * 100, 1) if profit_base > 0 else float("nan"),
        "peer_rows":       peer_rows,
        "peer_unis":       peer_unis,
        "ci_width_pct":    round(ci_width_rel * 100, 1),
        "edge_hit":        edge_hit,
        "unstable":        unstable,
        "status":          publish,
    }


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    # Короткі аліаси університетів для зручності в CLI
    UNI_ALIASES = {
        "накума": 'Києво-Могилянська',
        "кма":    'Києво-Могилянська',
        "уку":    'католицький',
        "auk":    'Американ',
        "step":   'STEP',
        "кse":    'Київська школа',
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="KSE Bachelor Pricing Optimization"
    )
    parser.add_argument("--csv",     required=True,  help="Шлях до CSV")
    parser.add_argument("--mc-usd",  type=float, default=2000,   help="MC в USD (default: 2000)")
    parser.add_argument("--n-boot",  type=int,   default=N_BOOT, help="Кількість bootstrap-ітерацій")
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        metavar="РІК",
        help="Роки для регресії. Приклад: --years 2023 2024 2025"
    )
    parser.add_argument(
        "--exclude-unis", nargs="+", default=None,
        metavar="УНІ",
        help=(
            "Виключити peer-університети з регресії.\n"
            "Аліаси: накума/кма, уку, auk, step\n"
            "Приклад: --exclude-unis auk step"
        )
    )
    parser.add_argument(
        "--include-unis", nargs="+", default=None,
        metavar="УНІ",
        help=(
            "Залишити ТІЛЬКИ ці peer-університети (whitelist).\n"
            "Аліаси: накума/кма, укв, auk, step\n"
            "Приклад: --include-unis накума укu"
        )
    )
    parser.add_argument(
        "--kse-year", type=int, default=2025,
        metavar="РІК",
        help="Рік KSE-програм для оптимізації (default: 2025)"
    )
    args = parser.parse_args()

    MC = args.mc_usd * UAH_PER_USD

    # Розкрити аліаси → підрядки для str.contains
    def _resolve_unis(names):
        resolved = []
        for n in names:
            alias = UNI_ALIASES.get(n.lower())
            resolved.append(alias if alias else n)
        return resolved

    print(f"\n{'═'*65}")
    print(f"  KSE PRICING OPTIMIZATION  |  MC = ${args.mc_usd:,.0f} × {UAH_PER_USD} = {MC:,.0f} грн")
    print(f"{'═'*65}\n")

    # 1. Дані
    df = load_and_clean(args.csv)
    total_raw = len(df)

    # Фільтр років (застосовується до всього датасету включно з KSE)
    if args.years:
        df = df[df["рік"].isin(args.years)].copy()
        print(f"Фільтр років:    {sorted(args.years)}")

    # Фільтр peer-університетів (не чіпаємо KSE рядки)
    is_kse = df["університет_назва"].str.contains("Київська школа")
    peer_mask = ~is_kse

    if args.include_unis:
        unis = _resolve_unis(args.include_unis)
        keep = pd.Series(False, index=df.index)
        for u in unis:
            keep |= df["університет_назва"].str.contains(u, case=False)
        df = df[is_kse | (peer_mask & keep)].copy()
        print(f"Включено peers:  {args.include_unis} → {unis}")

    if args.exclude_unis:
        unis = _resolve_unis(args.exclude_unis)
        drop = pd.Series(False, index=df.index)
        for u in unis:
            drop |= df["університет_назва"].str.contains(u, case=False)
        df = df[~(peer_mask & drop)].copy()
        print(f"Виключено peers: {args.exclude_unis} → {unis}")

    print(f"Рядків у моделі: {len(df)}  (до фільтрів: {total_raw})")
    print(f"Унікальні роки:  {sorted(df['рік'].unique())}")
    peer_unis_list = df[~df['університет_назва'].str.contains('Київська школа')]['університет_назва'].unique()
    print(f"Peer-університети ({len(peer_unis_list)}): {', '.join(sorted(peer_unis_list))}\n")

    kse_all  = df[df["університет_назва"].str.contains("Київська школа")].copy()
    peer_all = df[~df["університет_назва"].str.contains("Київська школа")].copy()

    # 2. Регресія
    model, beta1, spec_cols, bp_pval, max_vif = fit_global_ols(df)

    print(f"\n── РЕГРЕСІЯ ──────────────────────────────────────────────────")
    print(f"  β₁ = {beta1:.7f} заяв/грн  ({beta1*1000:.4f} заяв / 1 000 грн)")
    print(f"  Adj. R² = {model.rsquared_adj:.4f}   N = {len(df)}")
    print(f"  p(β₁) = {model.pvalues['price_2026']:.4f}   (HC3 SE)")
    print(f"  Breusch–Pagan p = {bp_pval:.4f}  → {'гетероскедастичність, HC3 коректний' if bp_pval < 0.05 else 'гомоскедастичний'}")
    print(f"  max VIF (числові) = {max_vif:.2f}  → {'⚠ коліній.' if max_vif > 10 else 'OK'}")

    # 3. Bootstrap
    print(f"\n  Бутстреп {args.n_boot} ітерацій...", end=" ", flush=True)
    boot_betas = bootstrap_beta1(df, spec_cols, n_boot=args.n_boot)
    b10, b50, b90 = np.percentile(boot_betas, [10, 50, 90])
    print(f"готово.")
    print(f"  Bootstrap β₁: [{b10:.7f},  {b90:.7f}]  (median={b50:.7f})")

    # 4. ρ (all-years KSE)
    rho_by_spec, overall_rho = compute_rho_cascade(kse_all)

    print(f"\n── ρ ALL-YEARS KSE ───────────────────────────────────────────")
    for s, r in sorted(rho_by_spec.items()):
        print(f"  {s:<45} ρ = {r:.3f}")
    print(f"  {'[overall fallback]':<45} ρ = {overall_rho:.3f}")

    # 5. KSE programs for target year
    kse25 = df[
        df["університет_назва"].str.contains("Київська школа") &
        (df["рік"] == args.kse_year)
    ].copy()
    if kse25.empty:
        print(f"⚠ Немає KSE-рядків для року {args.kse_year}. Доступні роки: {sorted(df[df['університет_назва'].str.contains('Київська школа')]['рік'].unique())}")
        return

    # Історичний fullpay по програмі (всі роки KSE)
    hist_fp = {}
    for prog, grp in kse_all.groupby("освітня_програма"):
        hist_fp[prog] = {int(r["рік"]): int(r["fullpay"]) for _, r in grp.iterrows()}

    results = []
    for _, row in kse25.iterrows():
        spec = row["spec_group"]
        rho  = rho_by_spec.get(spec, overall_rho)

        peer_spec = peer_all[peer_all["spec_group"] == spec]
        peer_rows = len(peer_spec)
        peer_unis = peer_spec["університет_назва"].nunique()

        res = optimize_program(row, beta1, boot_betas, rho, MC, peer_rows, peer_unis)
        res["hist_fp"] = hist_fp.get(row["освітня_програма"], {})
        results.append(res)

    # 6. Вивід
    print(f"\n── РЕЗУЛЬТАТИ ────────────────────────────────────────────────")
    print(f"  MC = {MC:,.0f} грн  |  Profit = (p − MC) × fullpay(p)")
    print(f"  Baseline: model-consistent (ρ_all_years × apps_cur)")
    print(f"  fp_actual_2025 — фактичний набір 2025, тільки контекст\n")

    publishable = [r for r in results if r["status"] == "PUBLISH"]
    skipped     = [r for r in results if r["status"] != "PUBLISH"]

    for r in publishable:
        dir_arrow = "↑" if r["delta_p_pct"] > 0 else "↓"
        hist_str = "  ".join(
            f"{yr}: {cnt}" for yr, cnt in sorted(r["hist_fp"].items()) if yr != 2025
        )
        print(f"  {'─'*60}")
        print(f"  🟢 {r['Програма']}")
        print(f"     Ціна:      {r['p_fact_k']}k (факт) / {r['p_cur_k']}k (CPI→2026)  →  {r['p_opt_k']}k  ({dir_arrow}{abs(r['delta_p_pct'])}%)")
        print(f"     CI:        [{r['ci10_k']}k – {r['ci90_k']}k]   Lerner-check: {r['p_lerner_k']}k")
        print(f"     ε (попит):  {r['eps']}   ρ={r['rho_allyears']}")
        hist_prefix = f"{hist_str}  →  " if hist_str else ""
        print(f"     Платники:  {hist_prefix}[2025 факт: {r['fp_actual_2025']}]  →  [прогноз нова ціна: {r['fp_opt']}]")
        print(f"     Profit:    {r['profit_base_M']}M → {r['profit_opt_M']}M грн  (Δ{r['delta_profit_pct']:+.1f}%)")
        print(f"     Peers:     {r['peer_rows']} рядків / {r['peer_unis']} унів   CI_width={r['ci_width_pct']}%")

    if skipped:
        print(f"\n  {'─'*60}")
        print(f"  ПРОПУЩЕНО (не публікуємо):")
        for r in skipped:
            print(f"    🔴 {r['Програма']:45}  {r['status']}")

    # 7. Загальний підсумок по KSE
    pub_profit_base = sum(r["profit_base_M"] for r in publishable)
    pub_profit_opt  = sum(r["profit_opt_M"]  for r in publishable)
    all_profit_base = sum(r["profit_base_M"] for r in results)
    # для SKIP — profit не змінюємо (рекомендація відсутня → ціна стоїть)
    all_profit_opt  = sum(
        r["profit_opt_M"] if r["status"] == "PUBLISH" else r["profit_base_M"]
        for r in results
    )
    print(f"\n  {'═'*60}")
    print(f"  ЗАГАЛЬНИЙ ПІДСУМОК (model-consistent baseline)")
    print(f"  {'─'*60}")
    print(f"  {'Програма':<40} {'Факт 2025':>10} {'Базовий':>10} {'Оптим.':>10}")
    print(f"  {'─'*60}")
    for r in results:
        fact_profit = (r["p_cur_k"] * 1000 - MC) * r["fp_actual_2025"] / 1e6
        opt_val     = r["profit_opt_M"] if r["status"] == "PUBLISH" else r["profit_base_M"]
        mark        = "" if r["status"] == "PUBLISH" else " *"
        print(f"  {r['Програма']:<40} {fact_profit:>9.3f}M {r['profit_base_M']:>9.3f}M {opt_val:>9.3f}M{mark}")
    fact_total = sum((r["p_cur_k"]*1000 - MC) * r["fp_actual_2025"] / 1e6 for r in results)
    print(f"  {'─'*60}")
    print(f"  {'РАЗОМ (PUBLISH-програми)':<40} {'':>10} {pub_profit_base:>9.3f}M {pub_profit_opt:>9.3f}M")
    print(f"  {'РАЗОМ (всі програми)':<40} {fact_total:>9.3f}M {all_profit_base:>9.3f}M {all_profit_opt:>9.3f}M")
    print(f"  {'─'*60}")
    print(f"  * ціна не змінюється (SKIP) → оптим. = базовий")
    print(f"  Базовий = (p_cur − MC) × apps_cur × ρ_all_years")
    print(f"  Факт 2025 = (p_cur − MC) × fp_actual_2025")

    # 8. Summary table
    print(f"\n── ЗВЕДЕНА ТАБЛИЦЯ ───────────────────────────────────────────")
    cols = ["Програма", "p_fact_k", "p_cur_k", "p_opt_k", "delta_p_pct",
            "fp_actual_2025", "fp_model_base", "fp_opt",
            "ci10_k", "ci90_k", "eps", "delta_profit_pct", "status"]
    df_out = pd.DataFrame(results)[cols]
    df_out.columns = ["Програма", "p_факт_k", "p_cur_k", "p_opt_k", "Δp%",
                      "fp_факт25", "fp_базовий", "fp_прогноз",
                      "CI_p10", "CI_p90", "ε_попит", "Δprofit%", "status"]
    print(df_out.to_string(index=False))

    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()