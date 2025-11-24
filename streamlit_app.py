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
        {"name": "그룹 2", "days": ["화", "목", "토"]},
        {"name": "그룹 3", "days": ["일"]},
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
    df_sim = pd.read_csv(uploaded_file)

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

    # coef 계산
    df_sim["coef"] = df_sim["워크인"] / df_sim["예약"]

    # ========== Step 2: 요일 그룹핑 설정 ==========
    st.header("Step 2: 요일 그룹핑 설정")

    st.markdown(
        "각 그룹에 포함할 요일을 선택하세요. 회귀 분석 시 각 그룹이 독립변수로 사용됩니다."
    )

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

    for idx, col in enumerate(cols):
        with col:
            group = st.session_state.day_groups[idx]
            st.subheader(group["name"])

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

    # 그룹 추가/제거
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("그룹 추가"):
            new_group_num = len(st.session_state.day_groups) + 1
            st.session_state.day_groups.append(
                {"name": f"그룹 {new_group_num}", "days": []}
            )
            st.rerun()

    with col_b:
        if st.button("마지막 그룹 제거") and len(st.session_state.day_groups) > 1:
            st.session_state.day_groups.pop()
            st.rerun()

    # 그룹핑 요약
    st.markdown("**현재 그룹핑:**")
    for group in st.session_state.day_groups:
        if group["days"]:
            st.write(f"- {group['name']}: {', '.join(group['days'])}")

    # 요일 → 그룹 매핑 생성 및 검증
    day_to_group = {}
    duplicate_days = []

    for idx, group in enumerate(st.session_state.day_groups):
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

        # ========== 1. 성능 지표 (강조) ==========
        st.subheader("모델 성능 지표")

        # 큰 메트릭 카드로 표시
        st.markdown("### 시뮬레이션 데이터 (학습)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="RMSE",
                value=f"{metrics_sim['RMSE']:.2f}",
                help="Root Mean Squared Error: 예측 오차의 제곱근"
            )
        with col2:
            st.metric(
                label="NRMSE",
                value=f"{metrics_sim['NRMSE']:.3f}",
                help="Normalized RMSE: 평균 대비 상대적 오차"
            )
        with col3:
            st.metric(
                label="R² Score",
                value=f"{metrics_sim['R²']:.3f}",
                help="결정계수: 1에 가까울수록 좋음"
            )

        st.markdown("### 실제 데이터 (테스트) - 최종 평가")
        col1, col2, col3 = st.columns(3)
        with col1:
            delta_rmse = metrics_real['RMSE'] - metrics_sim['RMSE']
            st.metric(
                label="RMSE",
                value=f"{metrics_real['RMSE']:.2f}",
                delta=f"{delta_rmse:+.2f}",
                delta_color="inverse"
            )
        with col2:
            delta_nrmse = metrics_real['NRMSE'] - metrics_sim['NRMSE']
            st.metric(
                label="NRMSE",
                value=f"{metrics_real['NRMSE']:.3f}",
                delta=f"{delta_nrmse:+.3f}",
                delta_color="inverse"
            )
        with col3:
            delta_r2 = metrics_real['R²'] - metrics_sim['R²']
            st.metric(
                label="R² Score",
                value=f"{metrics_real['R²']:.3f}",
                delta=f"{delta_r2:+.3f}",
                delta_color="normal"
            )

        st.divider()

        # ========== 2. 회귀 계수 ==========
        st.subheader("회귀 계수")

        var_names = ["const"] + X_cols

        print("\n" + "="*60)
        print("회귀 계수 테이블 생성")
        print("="*60)
        print(f"변수 이름: {var_names}")
        print(f"X_cols: {X_cols}")
        print(f"계수 개수: {len(st.session_state.coef_with_intercept)}")
        print(f"변수 이름 개수: {len(var_names)}")

        coef_df = pd.DataFrame(
            {
                "변수": var_names,
                "계수": st.session_state.coef_with_intercept,
                "표준오차": st.session_state.se_coef,
                "t-value": st.session_state.t_values,
                "p-value": st.session_state.p_values,
            }
        )
        print("\n회귀 계수 테이블:")
        print(coef_df)
        print("="*60 + "\n")

        st.dataframe(coef_df, use_container_width=True, height=250)

        st.divider()

        # ========== 3. 시뮬레이션 데이터 분석 ==========
        st.subheader("시뮬레이션 데이터 분석")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_sim["워크인"], df_sim["pred_walkin"], alpha=0.6, s=20)
            mn, mx = df_sim["워크인"].min(), df_sim["워크인"].max()
            ax.plot([mn, mx], [mn, mx], "r--", label="Perfect", linewidth=1)
            ax.set_xlabel("Actual", fontsize=9)
            ax.set_ylabel("Predicted", fontsize=9)
            ax.set_title("Actual vs Predicted", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            # 상관관계 히트맵
            corr_cols = ["강수여부", "예약", "워크인", "coef"]
            corr_matrix = df_sim[corr_cols].corr()

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                       ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
            ax.set_title("상관관계 히트맵", fontsize=10)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Residuals (전체 너비)
        df_sim["residual"] = df_sim["워크인"] - df_sim["pred_walkin"]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(len(df_sim)), df_sim["residual"], alpha=0.7, width=1.0)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Index", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.set_title("시뮬레이션: Residuals", fontsize=10)
        ax.tick_params(labelsize=8)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()

        # ========== 4. 실제 데이터 평가 ==========
        st.subheader("실제 데이터 평가 (14개)")

        # 테이블
        display_cols = ["날짜", "요일", "강수여부", "예약", "워크인", "pred_walkin"]
        st.dataframe(
            df_real[display_cols].round(2), use_container_width=True, height=550
        )

        # 요일 순서 정의
        day_order = ["월", "화", "수", "목", "금", "토", "일"]

        # 2024년과 2025년 데이터 분리
        df_24_eval = df_real[df_real["날짜"].astype(str).str.contains("2024")].copy()
        df_25_eval = df_real[df_real["날짜"].astype(str).str.contains("2025")].copy()

        # 요일을 Categorical로 변환하여 순서 보장
        df_24_eval["요일"] = pd.Categorical(df_24_eval["요일"], categories=day_order, ordered=True)
        df_25_eval["요일"] = pd.Categorical(df_25_eval["요일"], categories=day_order, ordered=True)

        # 요일별로 정렬
        df_24_eval = df_24_eval.sort_values("요일")
        df_25_eval = df_25_eval.sort_values("요일")

        # Actual vs Predicted 산점도
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df_real["워크인"], df_real["pred_walkin"], s=80, alpha=0.7)
            mn, mx = df_real["워크인"].min(), df_real["워크인"].max()
            ax.plot([mn, mx], [mn, mx], "r--", label="Perfect", linewidth=1)
            ax.set_xlabel("Actual 워크인", fontsize=9)
            ax.set_ylabel("Predicted 워크인", fontsize=9)
            ax.set_title("Actual vs Predicted (전체)", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            # Residuals
            df_real["residual"] = df_real["워크인"] - df_real["pred_walkin"]
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ["red" if x < 0 else "green" for x in df_real["residual"]]
            ax.bar(range(len(df_real)), df_real["residual"], color=colors, alpha=0.7)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("Index", fontsize=9)
            ax.set_ylabel("Residual", fontsize=9)
            ax.set_title("Residuals (전체)", fontsize=10)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")
        st.markdown("**요일별 비교 (2024 vs 2025)**")

        # 요일별 선 그래프 (2024 vs 2025)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**2024년 요일별 추세**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x_pos = range(len(df_24_eval))
            ax.plot(x_pos, df_24_eval["워크인"], marker="o", label="Actual",
                   linewidth=1.5, markersize=6)
            ax.plot(x_pos, df_24_eval["pred_walkin"], marker="s", label="Predicted",
                   linewidth=1.5, markersize=6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_24_eval["요일"], fontsize=8)
            ax.set_ylabel("워크인", fontsize=9)
            ax.set_title("2024년 요일별", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("**2025년 요일별 추세**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x_pos = range(len(df_25_eval))
            ax.plot(x_pos, df_25_eval["워크인"], marker="o", label="Actual",
                   linewidth=1.5, markersize=6)
            ax.plot(x_pos, df_25_eval["pred_walkin"], marker="s", label="Predicted",
                   linewidth=1.5, markersize=6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_25_eval["요일"], fontsize=8)
            ax.set_ylabel("워크인", fontsize=9)
            ax.set_title("2025년 요일별", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # 요일별 막대 그래프 (2024 vs 2025)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**2024년 요일별 비교**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x = range(len(df_24_eval))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                df_24_eval["워크인"],
                width,
                label="Actual",
            )
            ax.bar(
                [i + width / 2 for i in x],
                df_24_eval["pred_walkin"],
                width,
                label="Predicted",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df_24_eval["요일"], fontsize=8)
            ax.set_ylabel("워크인 수", fontsize=9)
            ax.set_title("2024년 막대 비교", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("**2025년 요일별 비교**")
            fig, ax = plt.subplots(figsize=(5, 4))
            x = range(len(df_25_eval))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                df_25_eval["워크인"],
                width,
                label="Actual",
            )
            ax.bar(
                [i + width / 2 for i in x],
                df_25_eval["pred_walkin"],
                width,
                label="Predicted",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(df_25_eval["요일"], fontsize=8)
            ax.set_ylabel("워크인 수", fontsize=9)
            ax.set_title("2025년 막대 비교", fontsize=10)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

else:
    st.info("CSV 파일을 업로드하여 시작하세요.")
