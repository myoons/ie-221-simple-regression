import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


# -------------------------------------------------------------
# 한국어 폰트 설정
# -------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------
# 1. 원본 데이터 로드 (2024 + 2025 병합)
# -------------------------------------------------------------
df_24 = pd.read_excel("real_data.xlsx", sheet_name="2024")
df_25 = pd.read_excel("real_data.xlsx", sheet_name="2025")
df = pd.concat([df_24, df_25], ignore_index=True)

# -------------------------------------------------------------
# 2. 전처리
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

# -----------------------------
# (변경 1) 강수 여부 기준: 1mm 이상
# -----------------------------
df["rain_dummy"] = (df["rain"] >= 1.0).astype(int)

df["coef"] = df["워크인"] / df["예약"]

# -------------------------------------------------------------
# 3. 변수 설정 — 평균기온 + 평균운량 포함
# -------------------------------------------------------------
X_cols = ["rain_dummy", "weekend", "평균기온", "평균운량"]

# -------------------------------------------------------------
# 4. 부트스트랩 설정
# -------------------------------------------------------------
N_BOOT = 5000

coef_results_res = []
coef_results_coef = []

for _ in range(N_BOOT):
    sample = df.sample(n=len(df), replace=True)

    # 예약 회귀
    X_res = sm.add_constant(sample[X_cols])
    y_res = sample["예약"]
    model_res = sm.OLS(y_res, X_res).fit()
    coef_results_res.append(model_res.params.values)

    # coef 회귀
    X_cf = sm.add_constant(sample[X_cols])
    y_cf = sample["coef"]
    model_cf = sm.OLS(y_cf, X_cf).fit()
    coef_results_coef.append(model_cf.params.values)

coef_results_res = np.array(coef_results_res)
coef_results_coef = np.array(coef_results_coef)

# 자연어 출력용 이름 변경
name_map = {
    "const": "const",
    "rain_dummy": "강수 여부",
    "weekend": "주말 여부",
    "평균기온": "평균기온",
    "평균운량": "평균운량",
}

coef_names = ["const"] + X_cols

# -------------------------------------------------------------
# 5. 부트스트랩 계수 분포 시각화 (한국어 이름 적용)
# -------------------------------------------------------------
for i, name in enumerate(coef_names):
    plt.figure(figsize=(6, 4))
    sns.histplot(coef_results_res[:, i], kde=True)
    plt.title(f"[예약 회귀] {name_map[name]} 계수 분포 (부트스트랩 {N_BOOT}회)")
    plt.axvline(
        np.mean(coef_results_res[:, i]), color="red", linestyle="--", label="mean"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bootstrap/bootstrap_reservation_{name}.png")
    plt.close()

for i, name in enumerate(coef_names):
    plt.figure(figsize=(6, 4))
    sns.histplot(coef_results_coef[:, i], kde=True)
    plt.title(
        f"[walk-in 비율(coef) 회귀] {name_map[name]} 계수 분포 (부트스트랩 {N_BOOT}회)"
    )
    plt.axvline(
        np.mean(coef_results_coef[:, i]), color="red", linestyle="--", label="mean"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bootstrap/bootstrap_coef_{name}.png")
    plt.close()


# -------------------------------------------------------------
# 6. 계수 방향성 유지 비율 계산 (한국어 이름 적용)
# -------------------------------------------------------------
def sign_consistency(arr):
    positive = np.mean(arr > 0)
    negative = np.mean(arr < 0)
    return positive, negative


consistency_report = {}

# 예약 회귀
for i, name in enumerate(coef_names):
    pos, neg = sign_consistency(coef_results_res[:, i])
    consistency_report[name] = {
        "예약_양수비율": round(pos * 100, 1),
        "예약_음수비율": round(neg * 100, 1),
    }

# coef 회귀
for i, name in enumerate(coef_names):
    pos, neg = sign_consistency(coef_results_coef[:, i])
    consistency_report[name].update(
        {
            "walkin비율_양수비율": round(pos * 100, 1),
            "walkin비율_음수비율": round(neg * 100, 1),
        }
    )

# 콘솔 출력 (한국어 변수명)
print("\n===== 계수 방향성 신뢰도 (부트스트랩 {}회) =====".format(N_BOOT))
for name, rep in consistency_report.items():
    print(f"\n▶ {name_map[name]}")
    for k, v in rep.items():
        print(f"  - {k}: {v}%")
