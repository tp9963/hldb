import pandas as pd
import numpy as np
import json

# ========================
# 配置
# ========================
EXCEL_FILE = "data.xlsx"
SHEET_NAME = "Sheet1"
JSON_FILE = "data.json"

# 股息率分位阈值（可根据你的调整修改）
DIVIDEND_YIELD_PERCENTILES = {
    "danger": 25,        # 危险区上沿（股息率分位低于此值 = 太贵）
    "low": 40,           # 偏低区上沿
    "fair_high": 60,     # 合理区上沿
    "high": 75,          # 偏高区上沿
    # >75% = 机会值
}

# =========================================================
# 🎯 阴影条件调节开关 (根据需要修改这里即可)
# 逻辑说明：
# 严格模式（历史上极少出现）：分位 >= 80 & 跌破 -1σ
# 平衡模式（推荐，当前使用）：分位 >= 75 & 跌破 -1σ
# 宽松模式（阴影会比较多）：分位 >= 75 | 跌破 -1σ  (把 & 换成 |)
SHADOW_PERCENTILE_THRESHOLD = 75
SHADOW_MODE = "AND"  # 可选 "AND" 或 "OR"
# =========================================================

def log_linear_regression(x, y):
    """
    对 y 取对数后做线性回归（最小二乘法）
    y_log = a * x + b
    返回：拟合值（原始尺度）、残差标准差、±1σ 上下轨、±2σ 上下轨
    """
    y_log = np.log(y)
    n = len(x)

    # 最小二乘法求解 a, b
    sum_x = np.sum(x)
    sum_y = np.sum(y_log)
    sum_xy = np.sum(x * y_log)
    sum_xx = np.sum(x * x)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_y - a * sum_x) / n

    # 拟合值（对数尺度）
    y_log_fit = a * x + b

    # 残差（对数尺度）
    residuals = y_log - y_log_fit
    sigma_log = np.std(residuals, ddof=1)  # 样本标准差

    # 转回原始尺度
    y_fit = np.exp(y_log_fit)
    y_upper = np.exp(y_log_fit + sigma_log)
    y_lower = np.exp(y_log_fit - sigma_log)

    # 新增：±2σ 通道
    y_upper_2 = np.exp(y_log_fit + 2 * sigma_log)
    y_lower_2 = np.exp(y_log_fit - 2 * sigma_log)

    return y_fit, y_upper, y_lower, y_upper_2, y_lower_2, sigma_log


def compute_dividend_yield_percentile(series, dates, years=10, min_obs=250):
    """
    计算每个时间点的股息率历史分位，使用截至当天的最近 N 年滚动窗口。
    如果窗口内数据点少于 min_obs，则使用截至当天的全部历史数据兜底。
    """
    percentiles = []
    for i in range(len(series)):
        current_date = dates.iloc[i]
        start_date = current_date - pd.DateOffset(years=years)
        mask = (dates >= start_date) & (dates <= current_date)
        hist = series[mask]
        if len(hist) < min_obs:
            hist = series.iloc[: i + 1]
        pct = (hist < series.iloc[i]).sum() / len(hist) * 100
        percentiles.append(round(pct, 2))
    return percentiles


# ========================
# 1. 读取Excel
# ========================
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=1)
df.columns = [str(c).strip() for c in df.columns]

df = df.rename(columns={
    "Date": "date",
    "close": "close",
    "val_dividendyield3": "dividend_yield_ttm",
})

df = df[["date", "close", "dividend_yield_ttm"]].copy()

# 清洗数据
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["dividend_yield_ttm"] = pd.to_numeric(df["dividend_yield_ttm"], errors="coerce")

df = df.dropna(subset=["date", "close", "dividend_yield_ttm"])
df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

print(f"数据加载完成，共 {len(df)} 条，时间范围：{df['date'].min()} ~ {df['date'].max()}")

# ========================
# 2. 计算对数回归通道（包含 ±1σ 和 ±2σ）
# ========================
x = np.arange(len(df), dtype=float)
y = df["close"].values

reg_mid, reg_upper, reg_lower, reg_upper_2, reg_lower_2, sigma_log = log_linear_regression(x, y)

df["reg_mid"] = reg_mid
df["reg_upper"] = reg_upper
df["reg_lower"] = reg_lower
df["reg_upper_2"] = reg_upper_2
df["reg_lower_2"] = reg_lower_2

# 相对中轨的σ偏移值
df["sigma_offset"] = np.log(y) - np.log(reg_mid)
df["sigma_offset"] = df["sigma_offset"].round(4)

print(f"对数回归通道计算完成，1σ = {sigma_log:.4f}（对数尺度）")

# ========================
# 3. 计算股息率历史分位（近10年滚动窗口）
# ========================
df["yield_percentile"] = compute_dividend_yield_percentile(
    df["dividend_yield_ttm"], df["date"], years=10
)

# ========================
# 4. 计算动态档位线（基于近10年数据）
# ========================
latest_yield = df["dividend_yield_ttm"].iloc[-1]
latest_close = df["close"].iloc[-1]
implied_dividend = latest_yield / 100 * latest_close

# 取最新日期往前推10年的股息率序列
latest_date = df["date"].iloc[-1]
ten_years_ago = latest_date - pd.DateOffset(years=10)
recent_mask = df["date"] >= ten_years_ago
recent_yield_series = df.loc[recent_mask, "dividend_yield_ttm"]
if len(recent_yield_series) < 250:
    recent_yield_series = df["dividend_yield_ttm"]  # 兜底

thresholds = {}
for key, pct in DIVIDEND_YIELD_PERCENTILES.items():
    yield_at_pct = np.percentile(recent_yield_series, pct)
    point_at_pct = implied_dividend / (yield_at_pct / 100)
    thresholds[key] = {
        "yield": round(yield_at_pct, 2),
        "point": round(point_at_pct, 2)
    }
    print(f"档位线 [{key}]：股息率 {yield_at_pct:.2f}% → 点位 {point_at_pct:.2f}")

df.attrs["thresholds"] = thresholds

# ========================
# 5. 标记条件阴影区间（可调节）
# ========================
if SHADOW_MODE == "AND":
    df["shadow_condition"] = (
        (df["yield_percentile"] >= SHADOW_PERCENTILE_THRESHOLD) & (df["close"] < df["reg_lower"])
    )
else:
    df["shadow_condition"] = (
        (df["yield_percentile"] >= SHADOW_PERCENTILE_THRESHOLD) | (df["close"] < df["reg_lower"])
    )

shadow_count = df["shadow_condition"].sum()
print(f"满足阴影条件的数据点：{shadow_count} 个（占总数的 {shadow_count/len(df)*100:.1f}%）")
if shadow_count == 0:
    print("⚠️ 警告：当前阴影条件过于严格，建议把 SHADOW_PERCENTILE_THRESHOLD 调低，或把 SHADOW_MODE 改成 'OR'")

# ========================
# 6. 构建输出JSON
# ========================
output_records = []
for _, row in df.iterrows():
    output_records.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "close": round(row["close"], 4),
        "dividend_yield_ttm": round(row["dividend_yield_ttm"], 4),
        "reg_mid": round(row["reg_mid"], 2),
        "reg_upper": round(row["reg_upper"], 2),
        "reg_lower": round(row["reg_lower"], 2),
        "reg_upper_2": round(row["reg_upper_2"], 2),
        "reg_lower_2": round(row["reg_lower_2"], 2),
        "sigma_offset": row["sigma_offset"],
        "yield_percentile": row["yield_percentile"],
        "shadow_condition": bool(row["shadow_condition"]),
    })

output = {
    "meta": {
        "index_code": "H30269",
        "index_name": "红利低波",
        "data_start": df["date"].min().strftime("%Y-%m-%d"),
        "data_end": df["date"].max().strftime("%Y-%m-%d"),
        "total_points": len(df),
        "sigma_log": round(sigma_log, 4),
        "latest": {
            "date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "close": round(df["close"].iloc[-1], 4),
            "dividend_yield_ttm": round(df["dividend_yield_ttm"].iloc[-1], 4),
            "yield_percentile": df["yield_percentile"].iloc[-1],
            "reg_mid": round(df["reg_mid"].iloc[-1], 2),
            "reg_upper": round(df["reg_upper"].iloc[-1], 2),
            "reg_lower": round(df["reg_lower"].iloc[-1], 2),
            "reg_upper_2": round(df["reg_upper_2"].iloc[-1], 2),
            "reg_lower_2": round(df["reg_lower_2"].iloc[-1], 2),
        },
        "thresholds": thresholds,
        "dividend_yield_percentile_config": DIVIDEND_YIELD_PERCENTILES,
    },
    "data": output_records,
}

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 完成！共 {len(output_records)} 条，已生成 {JSON_FILE}")