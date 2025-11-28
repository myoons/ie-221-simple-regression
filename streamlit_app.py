import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# 한글 폰트 설정
import matplotlib as mpl

# CRITICAL: sns.set()을 먼저 호출해야 함 (폰트 설정을 초기화하므로)
sns.set(style="whitegrid", font_scale=1.0)

# sns.set() 이후에 폰트 설정 (이게 핵심!)
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# 추가 보험: rc_context에서도 동일하게 설정
plt.rc('font', family='Apple SD Gothic Neo')

st.set_page_config(page_title="워크인 예측 회귀 분석", layout="wide")

# 타이틀
st.title("워크인 예측 회귀 분석 시스템")
st.markdown("""
**목적**: 시뮬레이션 데이터로 회귀 모델을 학습하고 실제 데이터로 평가합니다.
- 종속변수: 워크인/예약 비율 (coef)
- 독립변수: 강수여부, 요일 그룹
""")

# Session state 초기화
if "day_groups" not in st.session_state:
    st.session_state.day_groups = [
        {"name": "그룹 1", "days": ["월", "수", "금"]},
        {"name": "그룹 2", "days": ["화", "목", "토", "일"]},
    ]

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ========== Step 1: CSV 업로드 ==========
st.header("Step 1: 시뮬레이션 데이터 업로드")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요 (포맷: 요일,강수여부,예약,워크인)",
        type=["csv"],
        help="샘플: sample_simulation.csv",
    )

with col2:
    st.markdown("**필수 컬럼:**")
    st.code("요일,강수여부,예약,워크인")
    with open("sample_simulation.csv", "rb") as f:
        st.download_button(
            "샘플 CSV 다운로드",
            f,
            file_name="sample_simulation.csv",
            mime="text/csv",
        )

if uploaded_file is not None:
    df_sim = pd.read_csv(uploaded_file, encoding='utf-8-sig')

    # 데이터 검증
    required_cols = ["요일", "강수여부", "예약", "워크인"]
    if not all(col in df_sim.columns for col in required_cols):
        st.error(f"CSV 파일에 필수 컬럼이 없습니다: {required_cols}")
        st.stop()

    st.success(f"[완료] 데이터 로드 완료: {len(df_sim)}개 행")

    with st.expander("데이터 미리보기"):
        st.dataframe(df_sim.head(20), use_container_width=True)
        st.write(f"**통계:**")
        st.write(df_sim.describe())

    # 예약이 0인 행 필터링 (coef 계산 시 division by zero 방지)
    zero_reservation_count = (df_sim["예약"] == 0).sum()
    if zero_reservation_count > 0:
        st.warning(f"[주의] 예약이 0인 행 {zero_reservation_count}개가 분석에서 제외됩니다.")
        df_sim = df_sim[df_sim["예약"] > 0].reset_index(drop=True)

    # coef 계산
    df_sim["coef"] = df_sim["워크인"] / df_sim["예약"]

    # ========== Step 2: 요일 그룹핑 설정 ==========
    st.header("Step 2: 요일 그룹핑 설정")

    col_header_left, col_header_right = st.columns([3, 1])
    with col_header_left:
        st.markdown(
            "각 그룹에 포함할 요일을 선택하세요. 회귀 분석 시 각 그룹이 독립변수로 사용됩니다."
        )
    with col_header_right:
        if st.button("그룹 추가", type="secondary", use_container_width=True):
            new_group_num = len(st.session_state.day_groups) + 1
            st.session_state.day_groups.append(
                {"name": f"그룹 {new_group_num}", "days": []}
            )
            st.rerun()

    all_days = ["월", "화", "수", "목", "금", "토", "일"]

    # 그룹 편집 UI - 체크박스 상태 관리

    # 변경 감지를 위한 플래그
    if "prev_day_groups" not in st.session_state:
        st.session_state.prev_day_groups = [g.copy() for g in st.session_state.day_groups]

    cols = st.columns(len(st.session_state.day_groups))

    # 현재 선택 상태 파악
    current_selections = {}
    for idx, group in enumerate(st.session_state.day_groups):
        current_selections[idx] = set(group["days"])

    # 새로운 선택 상태 저장
    new_selections = {idx: [] for idx in range(len(st.session_state.day_groups))}
    needs_rerun = False

    # 그룹 삭제를 위한 플래그
    group_to_delete = None

    for idx, col in enumerate(cols):
        with col:
            group = st.session_state.day_groups[idx]

            # 그룹 헤더와 삭제 버튼
            col_title, col_delete = st.columns([3, 1])
            with col_title:
                st.subheader(group["name"])
            with col_delete:
                if len(st.session_state.day_groups) > 1:
                    if st.button("🗑️", key=f"delete_group_{idx}", help="그룹 삭제"):
                        group_to_delete = idx

            for day in all_days:
                # 다른 그룹에 속해있는지 체크
                is_selected_elsewhere = False
                for other_idx, other_days in current_selections.items():
                    if other_idx != idx and day in other_days:
                        is_selected_elsewhere = True
                        break

                # 현재 체크 상태
                is_checked = day in current_selections[idx]

                # 체크박스 렌더링
                checkbox = st.checkbox(
                    day,
                    value=is_checked,
                    key=f"day_{idx}_{day}",
                    disabled=is_selected_elsewhere
                )

                # 체크박스 선택 시 추가
                if checkbox:
                    new_selections[idx].append(day)

    # 그룹 삭제 처리
    if group_to_delete is not None:
        st.session_state.day_groups.pop(group_to_delete)
        st.rerun()

    # session_state 업데이트 및 변경 감지
    for idx in range(len(st.session_state.day_groups)):
        old_days = set(st.session_state.day_groups[idx]["days"])
        new_days = set(new_selections[idx])

        # 변경 사항이 있으면 플래그 설정
        if old_days != new_days:
            needs_rerun = True

        st.session_state.day_groups[idx]["days"] = new_selections[idx]

    # 변경 사항이 있으면 즉시 rerun
    if needs_rerun:
        st.session_state.prev_day_groups = [g.copy() for g in st.session_state.day_groups]
        st.rerun()

    # 그룹핑 요약
    st.markdown("**현재 그룹핑:**")
    for group in st.session_state.day_groups:
        if group["days"]:
            st.write(f"- {group['name']}: {', '.join(group['days'])}")

    # 요일 → 그룹 매핑 생성 및 검증
    day_to_group = {}
    duplicate_days = []
    empty_groups = []

    for idx, group in enumerate(st.session_state.day_groups):
        if not group["days"]:
            # 빈 그룹 발견
            empty_groups.append(group["name"])
        for day in group["days"]:
            if day in day_to_group:
                # 중복 발견
                duplicate_days.append(day)
            else:
                day_to_group[day] = idx

    # 할당되지 않은 요일 체크
    ungrouped_days = [day for day in all_days if day not in day_to_group]

    # 에러 검증
    has_errors = False

    if empty_groups:
        st.error(f"[오류] **요일이 선택되지 않은 빈 그룹:** {', '.join(empty_groups)}")
        has_errors = True

    if ungrouped_days:
        st.error(f"[오류] **그룹에 할당되지 않은 요일:** {', '.join(ungrouped_days)}")
        has_errors = True

    if duplicate_days:
        st.error(f"[오류] **중복 할당된 요일:** {', '.join(set(duplicate_days))}")
        has_errors = True

    if not has_errors:
        st.success("[완료] 모든 요일이 올바르게 할당되었습니다.")

    # ========== Step 3: 회귀 분석 ==========
    st.header("Step 3: 회귀 분석 실행")

    # 에러가 있으면 버튼 비활성화
    if st.button("분석 시작", type="primary", use_container_width=True, disabled=has_errors):
        if has_errors:
            st.error("요일 그룹핑을 올바르게 설정해주세요.")
            st.stop()

        with st.spinner("회귀 분석 중..."):
            # One-Hot Encoding
            num_groups = len(st.session_state.day_groups)

            print("\n" + "="*60)
            print("회귀 분석 디버깅 로그")
            print("="*60)
            print(f"총 그룹 개수: {num_groups}")
            print(f"요일 → 그룹 매핑: {day_to_group}")

            for idx in range(num_groups):
                df_sim[f"group_{idx}"] = df_sim["요일"].apply(
                    lambda x: 1 if day_to_group.get(x) == idx else 0
                )
                group_count = df_sim[f"group_{idx}"].sum()
                print(f"group_{idx} 인코딩 완료: {group_count}개 데이터")

            # 독립변수 준비 (첫 번째 그룹은 기준으로 제외)
            X_cols = ["강수여부"] + [f"group_{i}" for i in range(1, num_groups)]
            print(f"\n독립변수 컬럼: {X_cols}")
            print(f"첫 번째 그룹(group_0)은 기준(reference)으로 제외됨")

            X_sim = df_sim[X_cols].values
            y_sim = df_sim["coef"].values

            print(f"\nX_sim shape: {X_sim.shape}")
            print(f"y_sim shape: {y_sim.shape}")
            print(f"\nX_sim 샘플 (처음 5행):")
            print(df_sim[X_cols].head())

            # 회귀 분석
            model = LinearRegression()
            model.fit(X_sim, y_sim)

            print(f"\n회귀 계수:")
            print(f"  intercept: {model.intercept_:.4f}")
            for i, col in enumerate(X_cols):
                print(f"  {col}: {model.coef_[i]:.4f}")

            # p-value 계산을 위한 추가 통계
            n = len(y_sim)
            k = X_sim.shape[1]
            y_pred_train = model.predict(X_sim)
            residuals = y_sim - y_pred_train
            mse = np.sum(residuals**2) / (n - k - 1)

            # 계수의 표준오차 계산
            X_with_const = np.column_stack([np.ones(n), X_sim])
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se_coef = np.sqrt(var_coef)

            # t-값과 p-값 계산
            coef_with_intercept = np.concatenate([[model.intercept_], model.coef_])
            t_values = coef_with_intercept / se_coef
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - k - 1))

            # 예측
            df_sim["pred_coef"] = model.predict(df_sim[X_cols].values)
            df_sim["pred_walkin"] = df_sim["예약"] * df_sim["pred_coef"]

            # 성능 지표
            rmse_sim = np.sqrt(mean_squared_error(df_sim["워크인"], df_sim["pred_walkin"]))
            nrmse_sim = rmse_sim / df_sim["워크인"].mean()
            r2_sim = r2_score(df_sim["워크인"], df_sim["pred_walkin"])

            # 실제 데이터 로드
            df_24 = pd.read_excel("real_data.xlsx", sheet_name="2024")
            df_25 = pd.read_excel("real_data.xlsx", sheet_name="2025")
            df_real = pd.concat([df_24, df_25], ignore_index=True)

            # 강수 여부 전처리
            def parse_rain(x):
                if isinstance(x, str):
                    x = x.strip()
                    if x == "-" or x == "":
                        return 0
                    if x.endswith("mm"):
                        x = x[:-2]
                    try:
                        return 1 if float(x) >= 1.0 else 0
                    except:
                        return 0
                return 1 if float(x) >= 1.0 else 0

            df_real["강수여부"] = df_real["일강수량"].apply(parse_rain)

            # 실제 데이터에 그룹핑 적용
            print("\n" + "="*60)
            print("실제 데이터에 그룹핑 적용")
            print("="*60)
            print(f"동일한 day_to_group 매핑 사용: {day_to_group}")

            for idx in range(num_groups):
                df_real[f"group_{idx}"] = df_real["요일"].apply(
                    lambda x: 1 if day_to_group.get(x) == idx else 0
                )
                group_count = df_real[f"group_{idx}"].sum()
                print(f"실제 데이터 group_{idx}: {group_count}개")

            print(f"\n실제 데이터 그룹핑 샘플:")
            print(df_real[["요일", "강수여부"] + [f"group_{i}" for i in range(num_groups)]].head(14))

            X_real = df_real[X_cols].values
            df_real["pred_coef"] = model.predict(X_real)
            df_real["pred_walkin"] = df_real["예약"] * df_real["pred_coef"]

            # 실제 데이터 성능 지표
            rmse_real = np.sqrt(mean_squared_error(df_real["워크인"], df_real["pred_walkin"]))
            nrmse_real = rmse_real / df_real["워크인"].mean()
            r2_real = r2_score(df_real["워크인"], df_real["pred_walkin"])

            # Session state에 저장
            st.session_state.model = model
            st.session_state.df_sim = df_sim
            st.session_state.df_real = df_real
            st.session_state.metrics_sim = {
                "RMSE": rmse_sim,
                "NRMSE": nrmse_sim,
                "R²": r2_sim,
            }
            st.session_state.metrics_real = {
                "RMSE": rmse_real,
                "NRMSE": nrmse_real,
                "R²": r2_real,
            }
            st.session_state.X_cols = X_cols
            st.session_state.coef_with_intercept = coef_with_intercept
            st.session_state.se_coef = se_coef
            st.session_state.t_values = t_values
            st.session_state.p_values = p_values
            st.session_state.analysis_done = True

        st.success("[완료] 분석 완료!")
        st.rerun()

    # ========== Step 4: 결과 시각화 ==========
    if st.session_state.analysis_done:
        st.header("Step 4: 결과 시각화")

        model = st.session_state.model
        df_sim = st.session_state.df_sim
        df_real = st.session_state.df_real
        metrics_sim = st.session_state.metrics_sim
        metrics_real = st.session_state.metrics_real
        X_cols = st.session_state.X_cols

        # ========== 1. 회귀 모델 (PPT 슬라이드 3) ==========
        st.subheader("회귀 모델")

        # 회귀식
        coef_intercept = model.intercept_
        coef_rain = model.coef_[0]
        coef_daygroup = model.coef_[1] if len(model.coef_) > 1 else 0

        st.markdown("### Model")
        st.latex(f"coef = {coef_intercept:.4f} + ({coef_rain:.4f}) \\times Rain + ({coef_daygroup:.4f}) \\times DayGroup")

        # Interpretation
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Interpretation")
            st.markdown(f"""
            - **Base ratio** (Mon/Wed/Fri, No Rain): **{coef_intercept*100:.1f}%**
            - **Rain effect**: **{coef_rain*100:+.1f}%p**
            - **Tue/Thu/Sat/Sun effect**: **{coef_daygroup*100:+.1f}%p**
            """)

        with col2:
            st.markdown("### Coefficients Table")
            var_names = ["Intercept", "Rain", "DayGroup"]
            coef_df = pd.DataFrame({
                "Variable": var_names[:len(st.session_state.coef_with_intercept)],
                "Coef": st.session_state.coef_with_intercept,
                "p-value": st.session_state.p_values,
            })
            st.dataframe(coef_df.round(4), use_container_width=True, hide_index=True)

        st.divider()

        # ========== 2. 성능 지표 ==========
        st.subheader("모델 성능 지표")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 시뮬레이션 데이터 (학습)")
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric(label="RMSE", value=f"{metrics_sim['RMSE']:.2f}")
            with subcol2:
                st.metric(label="NRMSE", value=f"{metrics_sim['NRMSE']:.1%}")
            with subcol3:
                st.metric(label="R²", value=f"{metrics_sim['R²']:.3f}")

        with col2:
            st.markdown("### 실제 데이터 (테스트)")
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric(label="RMSE", value=f"{metrics_real['RMSE']:.2f}")
            with subcol2:
                st.metric(label="NRMSE", value=f"{metrics_real['NRMSE']:.1%}")
            with subcol3:
                st.metric(label="R²", value=f"{metrics_real['R²']:.3f}")

        st.divider()

        # ========== 3. 예측 결과 시각화 (PPT 슬라이드 4) ==========
        st.subheader("예측 결과 시각화")

        # 요일 영어 변환
        day_to_eng = {"월": "Mon", "화": "Tue", "수": "Wed", "목": "Thu", "금": "Fri", "토": "Sat", "일": "Sun"}
        df_real["Day_eng"] = df_real["요일"].map(day_to_eng)
        day_order_eng = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # 3개 차트 (영어)
        col1, col2, col3 = st.columns(3)

        with col1:
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_real["워크인"], df_real["pred_walkin"], s=80, alpha=0.7, color='#3498DB', edgecolor='white', linewidth=1)
            mn, mx = min(df_real["워크인"].min(), df_real["pred_walkin"].min()), max(df_real["워크인"].max(), df_real["pred_walkin"].max())
            ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Walk-in', fontsize=10)
            ax.set_ylabel('Predicted Walk-in', fontsize=10)
            ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            # Residuals
            fig, ax = plt.subplots(figsize=(5, 4))
            residuals_real = df_real["워크인"] - df_real["pred_walkin"]
            colors = ['#E74C3C' if x < 0 else '#27AE60' for x in residuals_real]
            ax.bar(range(len(residuals_real)), residuals_real, color=colors, alpha=0.7, edgecolor='white')
            ax.axhline(0, color='black', linewidth=1)
            ax.set_xlabel('Index', fontsize=10)
            ax.set_ylabel('Residual', fontsize=10)
            ax.set_title('Residuals', fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col3:
            # Walk-in by Day
            fig, ax = plt.subplots(figsize=(5, 4))
            df_real["Day_order"] = pd.Categorical(df_real["Day_eng"], categories=day_order_eng, ordered=True)
            day_summary = df_real.groupby("Day_order").agg({
                "워크인": "mean",
                "pred_walkin": "mean"
            }).reset_index()

            x = range(len(day_summary))
            ax.plot(x, day_summary["워크인"], marker='o', linewidth=2, markersize=8, color='#3498DB', label='Actual')
            ax.plot(x, day_summary["pred_walkin"], marker='s', linewidth=2, markersize=8, color='#E74C3C', label='Predicted')
            ax.set_xticks(x)
            ax.set_xticklabels(day_order_eng, fontsize=9)
            ax.set_xlabel('Day of Week', fontsize=10)
            ax.set_ylabel('Walk-in (avg)', fontsize=10)
            ax.set_title('Walk-in by Day', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ========== 4. 조건별 예측 coef ==========
        st.subheader("조건별 예측 워크인 비율")

        pred_table = pd.DataFrame({
            "Condition": [
                "Mon/Wed/Fri + No Rain",
                "Mon/Wed/Fri + Rain",
                "Tue/Thu/Sat/Sun + No Rain",
                "Tue/Thu/Sat/Sun + Rain"
            ],
            "Predicted coef": [
                f"{coef_intercept:.3f} ({coef_intercept*100:.1f}%)",
                f"{coef_intercept + coef_rain:.3f} ({(coef_intercept + coef_rain)*100:.1f}%)",
                f"{coef_intercept + coef_daygroup:.3f} ({(coef_intercept + coef_daygroup)*100:.1f}%)",
                f"{coef_intercept + coef_rain + coef_daygroup:.3f} ({(coef_intercept + coef_rain + coef_daygroup)*100:.1f}%)"
            ]
        })
        st.dataframe(pred_table, use_container_width=True, hide_index=True)

        st.divider()

        # ========== 5. 실제 데이터 상세 ==========
        st.subheader("실제 데이터 상세 (14개)")

        # 테이블
        display_cols = ["날짜", "요일", "강수여부", "예약", "워크인", "pred_walkin"]
        st.dataframe(
            df_real[display_cols].round(2), use_container_width=True, height=400
        )

        # 요일 순서 정의 (영어)
        day_order = ["월", "화", "수", "목", "금", "토", "일"]

        # 2024년과 2025년 데이터 분리
        df_24_eval = df_real[df_real["날짜"].astype(str).str.contains("2024")].copy()
        df_25_eval = df_real[df_real["날짜"].astype(str).str.contains("2025")].copy()

        # 요일 영어 변환 추가
        df_24_eval["Day_eng"] = df_24_eval["요일"].map(day_to_eng)
        df_25_eval["Day_eng"] = df_25_eval["요일"].map(day_to_eng)

        # 요일을 Categorical로 변환하여 순서 보장
        df_24_eval["요일"] = pd.Categorical(df_24_eval["요일"], categories=day_order, ordered=True)
        df_25_eval["요일"] = pd.Categorical(df_25_eval["요일"], categories=day_order, ordered=True)

        # 요일별로 정렬
        df_24_eval = df_24_eval.sort_values("요일")
        df_25_eval = df_25_eval.sort_values("요일")

        st.markdown("---")
        st.markdown("**연도별 비교 (2024 vs 2025)**")

        # 요일별 선 그래프 (2024 vs 2025) - 영어
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**2024**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x_pos = range(len(df_24_eval))
            ax.plot(x_pos, df_24_eval["워크인"], marker="o", label="Actual",
                   linewidth=2, markersize=7, color='#3498DB')
            ax.plot(x_pos, df_24_eval["pred_walkin"], marker="s", label="Predicted",
                   linewidth=2, markersize=7, color='#E74C3C')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_24_eval["Day_eng"], fontsize=9)
            ax.set_ylabel("Walk-in", fontsize=10)
            ax.set_xlabel("Day of Week", fontsize=10)
            ax.set_title("2024 Walk-in by Day", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("**2025**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x_pos = range(len(df_25_eval))
            ax.plot(x_pos, df_25_eval["워크인"], marker="o", label="Actual",
                   linewidth=2, markersize=7, color='#3498DB')
            ax.plot(x_pos, df_25_eval["pred_walkin"], marker="s", label="Predicted",
                   linewidth=2, markersize=7, color='#E74C3C')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_25_eval["Day_eng"], fontsize=9)
            ax.set_ylabel("Walk-in", fontsize=10)
            ax.set_xlabel("Day of Week", fontsize=10)
            ax.set_title("2025 Walk-in by Day", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # 요일별 막대 그래프 (2024 vs 2025) - 영어
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            x = range(len(df_24_eval))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                df_24_eval["워크인"],
                width,
                label="Actual",
                color='#3498DB',
                alpha=0.8
            )
            ax.bar(
                [i + width / 2 for i in x],
                df_24_eval["pred_walkin"],
                width,
                label="Predicted",
                color='#E74C3C',
                alpha=0.8
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df_24_eval["Day_eng"], fontsize=9)
            ax.set_ylabel("Walk-in", fontsize=10)
            ax.set_title("2024 Comparison", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            x = range(len(df_25_eval))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                df_25_eval["워크인"],
                width,
                label="Actual",
                color='#3498DB',
                alpha=0.8
            )
            ax.bar(
                [i + width / 2 for i in x],
                df_25_eval["pred_walkin"],
                width,
                label="Predicted",
                color='#E74C3C',
                alpha=0.8
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df_25_eval["Day_eng"], fontsize=9)
            ax.set_ylabel("Walk-in", fontsize=10)
            ax.set_title("2025 Comparison", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

else:
    st.info("CSV 파일을 업로드하여 시작하세요.")
