import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------------
# 0. 기본 설정
# -------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "Apple SD Gothic Neo"  # 맥 기본 한글폰트
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("plots_final", exist_ok=True)

# -------------------------------------------------------------
# 1. 데이터 로드 (2024 + 2025)
# -------------------------------------------------------------
df_24 = pd.read_excel("real_data.xlsx", sheet_name="2024")
df_25 = pd.read_excel("real_data.xlsx", sheet_name="2025")
df = pd.concat([df_24, df_25], ignore_index=True)

# -------------------------------------------------------------
# 2. 전처리: 주말 여부, 강수 여부, coef
# -------------------------------------------------------------
df["weekend"] = df["요일"].isin(["토", "일"]).astype(int)


def parse_rain(x):
    if isinstance(x, str):
        x = x.strip()
        if x == "-" or x == "":
            return 0.0
        if x.endswith("mm"):
            x = x[:-2]
        try:
            return float(x)
        except:
            return 0.0
    return float(x)


df["rain"] = df["일강수량"].apply(parse_rain)
df["rain_dummy"] = (df["rain"] >= 1.0).astype(int)

# 예약이 0인 날이 있다면 coef 계산 시 문제되므로 방어
df = df[df["예약"] > 0].copy()
df["coef"] = df["워크인"] / df["예약"]

# -------------------------------------------------------------
# 3. 예약 회귀모델 (예약 ~ 강수여부 + 주말여부)
# -------------------------------------------------------------
X_res = sm.add_constant(df[["rain_dummy", "weekend"]])
y_res = df["예약"]

model_res = sm.OLS(y_res, X_res).fit()
df["pred_res"] = model_res.predict(X_res)

# -------------------------------------------------------------
# 4. coef 회귀모델 (coef ~ 강수여부 + 주말여부)
# -------------------------------------------------------------
X_cf = sm.add_constant(df[["rain_dummy", "weekend"]])
y_cf = df["coef"]

model_cf = sm.OLS(y_cf, X_cf).fit()
df["pred_coef"] = model_cf.predict(X_cf)

# 음수 coef 예측이 나오면 0으로 컷 (안정성용)
df["pred_coef"] = df["pred_coef"].clip(lower=0)

# -------------------------------------------------------------
# 5. 워크인 예측 (워크인 = 예약예측 × coef예측)
# -------------------------------------------------------------
df["pred_walkin"] = df["pred_res"] * df["pred_coef"]


# -------------------------------------------------------------
# 6. 성능 지표: RMSE + NRMSE
# -------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(a, b):
    return rmse(a, b) / float(np.mean(a))


rmse_res = rmse(df["예약"], df["pred_res"])
rmse_walkin = rmse(df["워크인"], df["pred_walkin"])
nrmse_res = nrmse(df["예약"], df["pred_res"])
nrmse_walkin = nrmse(df["워크인"], df["pred_walkin"])

print("===== 성능 지표 =====")
print(f"예약 RMSE   = {rmse_res:.2f}")
print(f"예약 NRMSE  = {nrmse_res:.3f} (평균 예약 대비 비율)")
print(f"워크인 RMSE = {rmse_walkin:.2f}")
print(f"워크인 NRMSE= {nrmse_walkin:.3f} (평균 워크인 대비 비율)")

# -------------------------------------------------------------
# 7. Prediction Interval (95%) - 예약 / coef
# -------------------------------------------------------------
pi_res = model_res.get_prediction(X_res).summary_frame(alpha=0.05)
df["res_pi_low"] = pi_res["obs_ci_lower"]
df["res_pi_high"] = pi_res["obs_ci_upper"]

pi_cf = model_cf.get_prediction(X_cf).summary_frame(alpha=0.05)
df["coef_pi_low"] = pi_cf["obs_ci_lower"]
df["coef_pi_high"] = pi_cf["obs_ci_upper"]

# -------------------------------------------------------------
# 8. 히트맵 (주말여부, 강수여부, 예약 수, 워크인 수)
# -------------------------------------------------------------
plt.figure(figsize=(6, 5))

# 원본 변수명 그대로 두되, 표시될 라벨만 변경
heat_df = df[["예약", "워크인", "rain_dummy", "weekend"]].copy()
heat_df = heat_df.rename(columns={"rain_dummy": "강수 여부", "weekend": "주말 여부"})

sns.heatmap(heat_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (예약 / 워크인 / 강수 여부 / 주말 여부)")
plt.tight_layout()
plt.savefig("plots_final/heatmap_basic.png")
plt.close()

# -------------------------------------------------------------
# 9. 예약 / 워크인 Actual vs Predicted + Residuals
# -------------------------------------------------------------
# 예약 Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(df["예약"], df["pred_res"])
mn, mx = df["예약"].min(), df["예약"].max()
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("Actual 예약")
plt.ylabel("Pred 예약")
plt.title("예약 Actual vs Predicted")
plt.tight_layout()
plt.savefig("plots_final/res_actual_pred.png")
plt.close()

# 예약 Residuals
df["res_residual"] = df["예약"] - df["pred_res"]
plt.figure(figsize=(8, 4))
plt.bar(df.index, df["res_residual"])
plt.axhline(0, color="black")
plt.title("예약 Residuals")
plt.xlabel("index")
plt.ylabel("잔차")
plt.tight_layout()
plt.savefig("plots_final/res_residuals.png")
plt.close()

# 워크인 Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(df["워크인"], df["pred_walkin"])
mn, mx = df["워크인"].min(), df["워크인"].max()
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("Actual 워크인")
plt.ylabel("Pred 워크인")
plt.title("워크인 Actual vs Predicted")
plt.tight_layout()
plt.savefig("plots_final/walkin_actual_pred.png")
plt.close()

# 워크인 Residuals
df["wk_residual"] = df["워크인"] - df["pred_walkin"]
plt.figure(figsize=(8, 4))
plt.bar(df.index, df["wk_residual"])
plt.axhline(0, color="black")
plt.title("워크인 Residuals")
plt.xlabel("index")
plt.ylabel("잔차")
plt.tight_layout()
plt.savefig("plots_final/walkin_residuals.png")
plt.close()

# -------------------------------------------------------------
# 10. 최적 계수 (const, 강수여부, 주말여부) 출력
# -------------------------------------------------------------
print("\n===== 예약 수 예측 회귀 계수 =====")
print(model_res.params)  # const, rain_dummy, weekend

print("\n===== 워크인 비율(coef) 예측 회귀 계수 =====")
print(model_cf.params)  # const, rain_dummy, weekend

# -------------------------------------------------------------
# 7-1. Prediction Interval Plot (예약)
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["날짜"], df["예약"], label="Actual 예약", marker="o")
plt.plot(df["날짜"], df["pred_res"], label="Pred 예약", marker="o")
plt.fill_between(
    df["날짜"],
    df["res_pi_low"],
    df["res_pi_high"],
    alpha=0.3,
    color="gray",
    label="95% Prediction Interval",
)
plt.title("예약 Prediction Interval (95%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("plots_final/reservation_prediction_interval.png")
plt.close()

# -------------------------------------------------------------
# 7-2. Prediction Interval Plot (coef)
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["날짜"], df["coef"], label="Actual coef", marker="o")
plt.plot(df["날짜"], df["pred_coef"], label="Pred coef", marker="o")
plt.fill_between(
    df["날짜"],
    df["coef_pi_low"],
    df["coef_pi_high"],
    alpha=0.3,
    color="gray",
    label="95% Prediction Interval",
)
plt.title("coef Prediction Interval (95%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("plots_final/coef_prediction_interval.png")
plt.close()


# -------------------------------------------------------------
# 11. 최종 예측 테이블 일부 확인
# -------------------------------------------------------------
print("\n===== 최종 예측 결과 (상위 몇 개) =====")
cols_show = [
    "날짜",
    "예약",
    "pred_res",
    "res_pi_low",
    "res_pi_high",
    "워크인",
    "pred_walkin",
    "coef",
    "pred_coef",
    "coef_pi_low",
    "coef_pi_high",
]
print(df[cols_show].head(20))
