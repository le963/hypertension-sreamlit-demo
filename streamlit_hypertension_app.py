# 高血压风险智能筛查系统 Streamlit 主程序
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

plt.ioff()
st.set_page_config(page_title="🩺高血压风险筛查系统", layout="wide")
st.title("🩺 高血压风险智能筛查与可解释分析系统")
st.markdown("全球校园人工智能算法精英大赛 | 算法创新赛道")

@st.cache_resource(show_spinner="正在加载模型...")
def load_model_files():
    model = joblib.load("rf_hypertension_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer

model, feature_cols, explainer = load_model_files()

INPUT_FEATURES = ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI", "Exercise_Level", "Family_History", "BP_History", "Smoking_Status"]

exercise_map_ui2val = {"低": 1, "中": 2, "高": 3}
family_map_ui2val = {"无高血压家族史": 0, "有高血压家族史": 1}
bp_map_ui2val = {"正常": "Normal", "高血压前期": "Prehypertension", "高血压": "Hypertension"}
bp_str2num = {"Normal":0, "Prehypertension":1, "Hypertension":2}
smoke_map_ui2val = {"没有吸烟习惯": "Non‑Smoker", "有吸烟习惯": "Smoker"}
smoke_str2num = {"Non‑Smoker":0, "Smoker":1}

# 和自动生成风险报告.py完全一致风险分层
def risk_level(p):
    if p < 0.3:
        return "低风险"
    elif p < 0.7:
        return "中风险"
    else:
        return "高风险"

def check_empty_rows(df):
    sub = df[INPUT_FEATURES].replace(["", np.nan, None], np.nan)
    all_null = sub.isna().all(axis=1)
    return ~all_null, list(df.index[all_null])

def preprocess_data(df_raw):
    df = df_raw.copy()
    df = df.fillna({
        "Age": 0, "Salt_Intake": 0, "Stress_Score": 0, "Sleep_Duration": 0, "BMI": 0,
        "Exercise_Level": 1, "Family_History": 0,
        "BP_History": "Normal", "Smoking_Status": "Non‑Smoker"
    })
    if "Exercise_Level" in df.columns and df["Exercise_Level"].dtype == object:
        df["Exercise_Level"] = df["Exercise_Level"].map(exercise_map_ui2val)
    if "Family_History" in df.columns and df["Family_History"].dtype == object:
        df["Family_History"] = df["Family_History"].map(family_map_ui2val)
    if "BP_History" in df.columns and df["BP_History"].dtype == object:
        df["BP_History"] = df["BP_History"].map(bp_map_ui2val).map(bp_str2num)
    if "Smoking_Status" in df.columns and df["Smoking_Status"].dtype == object:
        df["Smoking_Status"] = df["Smoking_Status"].map(smoke_map_ui2val).map(smoke_str2num)

    df = df.reindex(columns=feature_cols, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df[feature_cols]

def plot_shap_summary(X_data, df_model, max_show=100):
    n = X_data.shape[0]
    if n > max_show:
        X_data = X_data[:max_show]
        df_model = df_model.iloc[:max_show]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap_batch = explainer.shap_values(X_data)
    shap.summary_plot(shap_batch, df_model, feature_names=feature_cols, show=False)
    st.pyplot(fig)
    plt.close(fig)

# 单样本解析函数：高危(↑)，保护(↓)
def parse_single_result(X_row, raw_origin_bp_text, raw_origin_smoke_text):
    shap_values = explainer.shap_values(X_row)
    sample_shap = shap_values[0]
    shap_dict = dict(zip(feature_cols, sample_shap))
    shap_dict = {k: -v for k, v in shap_dict.items()}
    raw_data = X_row.iloc[0]
    risk_desc_list, protect_desc_list = [], []

    name_mapping = {
        "Age": "年龄",
        "Salt_Intake": "盐摄入量",
        "Stress_Score": "压力评分",
        "Sleep_Duration": "睡眠时长",
        "BMI": "BMI",
        "Exercise_Level": "运动水平",
        "Family_History":"家族病史",
        "BP_History": "血压病史等级",
        "Smoking_Status": "吸烟状态"
    }
    threshold = {
        "Age": 45,
        "Salt_Intake": 5,
        "Stress_Score": 5,
        "Sleep_Duration": 7,
        "BMI": 24
    }
    bad_desc_set = {"睡眠时长不足", "盐摄入量偏高", "盐摄入量偏低",
                    "BMI偏高（超重）", "BMI偏低（偏瘦）", "运动量不足","压力评分较高","年龄偏大"}
    good_desc_set = {"BMI处于正常区间", "睡眠时长充足", "运动水平较高", "运动水平中等"}

    for feat_name, shap_val in shap_dict.items():
        desc = ""
        is_binary_feature = False
        desc_is_bad = False

        if feat_name in ["Age","Salt_Intake","Stress_Score","Sleep_Duration","BMI","Exercise_Level"]:
            val = raw_data[feat_name]
            if feat_name == "Age":
                desc = "年龄偏大" if val > threshold["Age"] else "年龄偏小"
            elif feat_name == "Salt_Intake":
                if val > threshold["Salt_Intake"]:
                    desc = "盐摄入量偏高"
                elif val <3:
                    desc = "盐摄入量偏低"
                else:
                    desc = "盐摄入量处于正常范围"
            elif feat_name == "Stress_Score":
                desc = "压力评分较高" if val > threshold["Stress_Score"] else "压力评分较低"
            elif feat_name == "Sleep_Duration":
                desc = "睡眠时长不足" if val < threshold["Sleep_Duration"] else "睡眠时长充足"
            elif feat_name == "BMI":
                if val > threshold["BMI"]:
                    desc = "BMI偏高（超重）"
                    desc_is_bad = True
                elif val < 18.5:
                    desc = "BMI偏低（偏瘦）"
                    desc_is_bad = True
                else:
                    desc = "BMI处于正常区间"
                    desc_is_bad = False
            elif feat_name == "Exercise_Level":
                if val <=1:
                    desc = "运动量不足"
                    desc_is_bad = True
                elif val >=3:
                    desc = "运动水平较高"
                    desc_is_bad = False
                else:
                    desc = "运动水平中等"
                    desc_is_bad = False

            if abs(shap_val) > 0.01:
                if desc in bad_desc_set:
                    risk_desc_list.append(f"{desc}(↑)")
                elif desc in good_desc_set:
                    protect_desc_list.append(f"{desc}(↓)")

        elif feat_name == "Family_History":
            is_binary_feature = True
            val = raw_data[feat_name]
            desc = "存在高血压家族病史" if val == 1 else "无高血压家族病史"
            desc_is_bad = (val ==1)
        elif feat_name == "BP_History":
            is_binary_feature = True
            val = raw_data[feat_name]
            if val ==2:
                desc = "既往存在高血压病史"
            elif val ==1:
                desc = "既往处于高血压前期"
            else:
                desc = "既往自述血压正常"
            desc_is_bad = (val >=1)
        elif feat_name == "Smoking_Status":
            is_binary_feature = True
            val = raw_data[feat_name]
            desc = "有吸烟习惯" if val ==1 else "无吸烟习惯"
            desc_is_bad = (val ==1)

        if is_binary_feature:
            if abs(shap_val) > 0.01:
                if desc_is_bad:
                    risk_desc_list.append(f"{desc}(↑)")
                else:
                    protect_desc_list.append(f"{desc}(↓)")

    # 追加文本同步调换箭头
    bp_raw = raw_origin_bp_text
    if bp_raw == "高血压":
        risk_desc_list.append("既往存在高血压病史(↑)")
    elif bp_raw == "高血压前期":
        risk_desc_list.append("既往处于高血压前期(↑)")
    elif bp_raw == "正常":
        protect_desc_list.append("既往自述血压正常(↓)")

    smoke_raw = raw_origin_smoke_text
    if smoke_raw == "有吸烟习惯":
        risk_desc_list.append("有吸烟习惯(↑)")
    elif smoke_raw == "没有吸烟习惯":
        protect_desc_list.append("无吸烟习惯(↓)")

    risk_text = "、".join(risk_desc_list) if risk_desc_list else "无显著风险因素"
    protect_text = "、".join(protect_desc_list) if protect_desc_list else "无明显保护因素"
    return risk_text, protect_text, sample_shap


menu = st.sidebar.selectbox("功能菜单", ["单人风险预测", "👥多人在线录入分析", "📂CSV数据集上传分析"])

if menu == "单人风险预测":
    st.header("📝 个体体检指标录入")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄(10‑110岁)", min_value=10, max_value=110, value=45)
        salt = st.number_input("盐摄入量(0.0‑25.0g)", min_value=0.0, max_value=25.0, value=5.0)
        stress = st.number_input("压力评分(0‑10分)", min_value=0, max_value=10, value=5)
        sleep = st.number_input("睡眠时长(0.0‑16.0h)", min_value=0.0, max_value=16.0, value=7.0)
        bmi = st.number_input("BMI(12.0‑45.0)", min_value=12.0, max_value=45.0, value=23.0)
    with col2:
        exercise = st.selectbox("运动等级", options=[1, 2, 3], format_func=lambda x: {1: "低", 2: "中", 3: "高"}[x])
        family_history = st.selectbox("高血压家族史", options=[0, 1], format_func=lambda x: {0: "无", 1: "有"}[x])
        bp_history = st.selectbox("既往血压情况", options=["正常", "高血压前期", "高血压"])
        smoke_status = st.selectbox("吸烟状态", options=["没有吸烟习惯", "有吸烟习惯"])

    if st.button("🚀 开始风险评估"):
        with st.spinner("正在计算..."):
            person_data = {
                "Age": age,
                "Salt_Intake": salt,
                "Stress_Score": stress,
                "Sleep_Duration": sleep,
                "BMI": bmi,
                "Exercise_Level": exercise,
                "Family_History": family_history,
                "BP_History": bp_history,
                "Smoking_Status": smoke_status
            }
            df_input = preprocess_data(pd.DataFrame([person_data]))
            prob = model.predict_proba(df_input)[0, 1]
            risk_result = risk_level(prob)
            risk_text, protect_text, sample_shap = parse_single_result(df_input, bp_history, smoke_status)

            if risk_result == "低风险":
                advice = "建议维持现有健康生活习惯，每年进行一次血压体检。"
            elif risk_result == "中风险":
                advice = "建议定期监测血压，减少高盐饮食，控制体重，规律运动。"
            else:
                advice = "⚠️请尽快前往医院心内科完成相关检查，严格控盐，密切监测血压，遵从临床医生指导。"

        st.subheader("📋 风险筛查报告")
        st.write(f"**预估患病概率：{prob*100:.1f}%**")
        st.write(f"**风险等级：{risk_result}**")
        st.write(f"**主要风险因素：** {risk_text}")
        st.write(f"**保护因素：** {protect_text}")
        st.write(f"**健康建议：** {advice}")
        st.caption("本报告基于机器学习模型预测，仅作为流行病学健康筛查参考，不能替代临床医生诊断。")
        st.markdown("说明：(↑)代表流行病学高危因素（升高患病风险）；(↓)代表流行病学保护性因素（降低患病风险）。")

        st.subheader("🔍 特征贡献SHAP可视化")
        plot_shap_summary(df_input, df_input)


elif menu == "👥多人在线录入分析":
    st.header("👥 多人在线录入批量风险评估")
    if "multi_person_df" not in st.session_state:
        st.session_state.multi_person_df = pd.DataFrame([
            {
                "姓名": "人员A",
                "Age": 42.0,
                "Salt_Intake": 4.5,
                "Stress_Score": 4.0,
                "Sleep_Duration": 7.0,
                "BMI": 22.5,
                "Exercise_Level": "中",
                "Family_History": "无高血压家族史",
                "BP_History": "正常",
                "Smoking_Status": "没有吸烟习惯"
            },
            {
                "姓名": "人员B",
                "Age": 55.0,
                "Salt_Intake": 7.2,
                "Stress_Score": 7.0,
                "Sleep_Duration": 6.0,
                "BMI": 26.1,
                "Exercise_Level": "低",
                "Family_History": "有高血压家族史",
                "BP_History": "高血压前期",
                "Smoking_Status": "有吸烟习惯"
            }
        ])

    edited_df = st.data_editor(
        st.session_state.multi_person_df,
        num_rows="dynamic",
        use_container_width=True,
        key="multi_editor",
        column_config={
            "姓名": st.column_config.TextColumn("姓名", width=100),
            "Age": st.column_config.NumberColumn("年龄(10‑110岁)", min_value=10, max_value=110, step=1, width=145),
            "Salt_Intake": st.column_config.NumberColumn("盐摄入量(0.0‑25.0g)", min_value=0.0, max_value=25.0, step=0.1, width=180),
            "Stress_Score": st.column_config.NumberColumn("压力评分(0‑10分)", min_value=0, max_value=10, step=1, width=150),
            "Sleep_Duration": st.column_config.NumberColumn("睡眠时长(0.0‑16.0h)", min_value=0.0, max_value=16.0, step=0.5, width=170),
            "BMI": st.column_config.NumberColumn("BMI(12.0‑45.0)", min_value=12.0, max_value=45.0, step=0.1, width=145),
            "Exercise_Level": st.column_config.SelectboxColumn("运动等级", options=["低", "中", "高"], width=110),
            "Family_History": st.column_config.SelectboxColumn("家族史", options=["无高血压家族史", "有高血压家族史"], width=160),
            "BP_History": st.column_config.SelectboxColumn("既往血压", options=["正常", "高血压前期", "高血压"], width=180),
            "Smoking_Status": st.column_config.SelectboxColumn("吸烟", options=["没有吸烟习惯", "有吸烟习惯"], width=160),
        }
    )
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_batch = st.button("🚀 执行多员批量风险预测")
    with col_btn2:
        st.download_button(
            label="📥下载模板CSV",
            data=edited_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="多人录入模板.csv",
            mime="text/csv"
        )
    if run_batch:
        with st.spinner("正在批量计算..."):
            st.session_state.multi_person_df = edited_df
            df_batch_raw = edited_df.copy()
            valid_mask, invalid_rows = check_empty_rows(df_batch_raw)
            if len(invalid_rows) > 0:
                st.warning(f"⚠️空白行索引 {invalid_rows}，该行跳过预测。")
            df_valid_raw = df_batch_raw.loc[valid_mask].copy()
            if df_valid_raw.shape[0] == 0:
                st.error("❌无有效输入数据。")
                st.stop()
            df_model_input = preprocess_data(df_valid_raw.drop(columns=["姓名"]))
            batch_prob = model.predict_proba(df_model_input)[:, 1]
            output_df = df_valid_raw.copy()
            output_df["患病预测概率(%)"] = np.round(batch_prob * 100, 1)
            output_df["风险等级"] = [risk_level(p) for p in batch_prob]

        st.subheader("📊 多员批量预测结果总表")
        st.dataframe(output_df, use_container_width=True)
        st.download_button(
            label="📥下载预测结果CSV",
            data=output_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="多人批量预测结果.csv",
            mime="text/csv"
        )

        st.subheader("🔍全局SHAP特征贡献图（全部样本汇总）")
        plot_shap_summary(df_model_input, df_model_input)
        st.markdown("说明：(↑)代表流行病学高危因素（升高患病风险）；(↓)代表流行病学保护性因素（降低患病风险）。")

        st.markdown("### 📄展开查看每一位人员完整个人风险报告")
        for i, row in output_df.iterrows():
            with st.expander(f"【{row['姓名']}】概率：{row['患病预测概率(%)']}% | {row['风险等级']}"):
                single_X = df_model_input.iloc[i:i+1]
                risk_text, protect_text, shap_single = parse_single_result(
                    single_X, row["BP_History"], row["Smoking_Status"]
                )
                prob_i = batch_prob[i]
                res_level = risk_level(prob_i)
                if res_level == "低风险":
                    advice = "建议维持现有健康生活习惯，每年进行一次血压体检。"
                elif res_level == "中风险":
                    advice = "建议定期监测血压，减少高盐饮食，控制体重，规律运动。"
                else:
                    advice = "⚠️请尽快前往医院心内科完成相关检查，严格控盐，密切监测血压，遵从临床医生指导。"

                st.write(f"**预估患病概率：{prob_i*100:.1f}%**")
                st.write(f"**风险等级：{res_level}**")
                st.write(f"**主要风险因素：** {risk_text}")
                st.write(f"**保护因素：** {protect_text}")
                st.write(f"**健康建议：** {advice}")
                st.caption("本报告基于机器学习模型预测，仅作为流行病学健康筛查参考，不能替代临床医生诊断。")


elif menu == "📂CSV数据集上传分析":
    st.header("📂 上传CSV数据集批量风险评估")
    uploaded_csv = st.file_uploader("上传csv数据集", type="csv")
    if uploaded_csv is not None:
        df_batch = pd.read_csv(uploaded_csv, skip_blank_lines=True)
        df_batch = df_batch.dropna(how="all").reset_index(drop=True)
        st.subheader("数据预览")
        st.dataframe(df_batch.head(), use_container_width=True)
        st.write(f"原始样本量：{df_batch.shape[0]}")
        st.write(f"数据列名：{list(df_batch.columns)}")
        if st.button("批量运行模型预测"):
            with st.spinner("正在批量计算..."):
                missing_cols = [col for col in INPUT_FEATURES if col not in df_batch.columns]
                if missing_cols:
                    st.error(f"❌CSV缺少必要的特征列：{missing_cols}")
                    st.stop()
                valid_mask, invalid_rows = check_empty_rows(df_batch)
                if len(invalid_rows) > 0:
                    st.warning(f"⚠️过滤空样本，数量：{len(invalid_rows)}")
                df_valid = df_batch.loc[valid_mask].copy()
                if df_valid.shape[0] == 0:
                    st.error("❌CSV无有效数据。")
                    st.stop()
                df_batch_encode = preprocess_data(df_valid)
                batch_prob = model.predict_proba(df_batch_encode)[:, 1]
                result_df = df_valid.copy()
                result_df["患病预测概率(%)"] = np.round(batch_prob * 100, 1)
                result_df["风险等级"] = [risk_level(p) for p in batch_prob]

            st.subheader("📊 CSV批量预测结果总表")
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                label="📥下载批量预测结果",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="csv上传批量预测结果.csv",
                mime="text/csv"
            )

            st.subheader("🔍全局SHAP特征贡献图")
            plot_shap_summary(df_batch_encode, df_batch_encode)
            st.markdown("说明：(↑)代表流行病学高危因素（升高患病风险）；(↓)代表流行病学保护性因素（降低患病风险）。")

            st.markdown("### 📄展开查看CSV中每一条样本完整个人风险报告")
            for i, row in result_df.iterrows():
                with st.expander(f"【第{i+1}条样本】概率：{row['患病预测概率(%)']}% | {row['风险等级']}"):
                    single_X = df_batch_encode.iloc[i:i+1]
                    risk_text, protect_text, shap_single = parse_single_result(
                        single_X, row["BP_History"], row["Smoking_Status"]
                    )
                    prob_i = batch_prob[i]
                    res_level = risk_level(prob_i)
                    if res_level == "低风险":
                        advice = "建议维持现有健康生活习惯，每年进行一次血压体检。"
                    elif res_level == "中风险":
                        advice = "建议定期监测血压，减少高盐饮食，控制体重，规律运动。"
                    else:
                        advice = "⚠️请尽快前往医院心内科完成相关检查，严格控盐，密切监测血压，遵从临床医生指导。"

                    st.write(f"**预估患病概率：{prob_i*100:.1f}%**")
                    st.write(f"**风险等级：{res_level}**")
                    st.write(f"**主要风险因素：** {risk_text}")
                    st.write(f"**保护因素：** {protect_text}")
                    st.write(f"**健康建议：** {advice}")
                    st.caption("本报告基于机器学习模型预测，仅作为流行病学健康筛查参考，不能替代临床医生诊断。")
