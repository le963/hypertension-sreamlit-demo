# streamlit_hypertension_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

st.set_page_config(page_title="高血压风险筛查系统", layout="wide")
st.title("🩺 高血压风险智能筛查与可解释分析系统")
st.markdown("全球校园人工智能算法精英大赛 | 算法创新赛道")

@st.cache_resource
def load_model_files():
    model = joblib.load("rf_hypertension_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_cols, explainer

model, scaler, feature_cols, explainer = load_model_files()

INPUT_FEATURES = ["Age","Salt_Intake","Stress_Score","Sleep_Duration","BMI","Exercise_Level","Family_History","BP_History","Smoking_Status"]

def risk_level(p):
    if p < 0.3:
        return "低风险"
    elif p < 0.7:
        return "中风险"
    else:
        return "高风险"

def check_empty_rows(df):
    sub = df[INPUT_FEATURES].copy()
    sub = sub.replace(["", np.nan, None], np.nan)
    all_null = sub.isna().all(axis=1)
    invalid_idx = list(df.index[all_null])
    return ~all_null, invalid_idx

def preprocess_data(df_raw):
    df = df_raw.copy()
    df = pd.get_dummies(df, columns=["BP_History","Smoking_Status"], drop_first=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

def plot_shap(scale_data, df_model):
    fig, ax = plt.subplots(figsize=(10,6))
    shap_batch = explainer.shap_values(scale_data)
    shap.summary_plot(shap_batch[:,:,1], df_model, feature_names=feature_cols, show=False)
    st.pyplot(fig)

# ---------------- 侧边菜单 ----------------
menu = st.sidebar.selectbox(
    "功能菜单",
    [
        "单人风险预测",
        "👥多人在线录入分析",
        "📂CSV数据集上传分析"
    ]
)

# ================= 单人预测 =================
if menu == "单人风险预测":
    st.header("📝 个体体检指标录入")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=10, max_value=100, value=45)
        salt = st.number_input("盐摄入量", min_value=0.0, max_value=20.0, value=5.0)
        stress = st.number_input("压力评分", min_value=0, max_value=10, value=5)
        sleep = st.number_input("睡眠时长(小时)", min_value=0.0, max_value=12.0, value=7.0)
        bmi = st.number_input("BMI", min_value=14.0, max_value=40.0, value=23.0)
    with col2:
        exercise = st.selectbox("运动等级", options=[1,2,3], format_func=lambda x:{1:"低",2:"中",3:"高"}[x])
        family_history = st.selectbox("高血压家族史", options=[0,1], format_func=lambda x:{0:"无",1:"有"}[x])
        bp_history = st.selectbox("既往血压情况", options=["Normal","Prehypertension","Hypertension"])
        smoke_status = st.selectbox("吸烟状态", options=["Non‑Smoker","Smoker"])

    if st.button("🚀 开始风险评估"):
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
        df_input = pd.DataFrame([person_data])
        df_input = preprocess_data(df_input)
        input_scaled = scaler.transform(df_input)
        prob = model.predict_proba(input_scaled)[0,1]
        risk_result = risk_level(prob)

        shap_values = explainer.shap_values(input_scaled)
        sample_shap = shap_values[:,:,1][0]
        shap_dict = dict(zip(feature_cols, sample_shap))
        raw_data = df_input.iloc[0]

        risk_desc_list = []
        protect_desc_list = []
        threshold = {"Age": 45, "Age_low":25, "Salt_high":5.0,"Salt_low":3.0, "Stress_Score": 5, "Sleep_Duration":7, "BMI":24}
        feature_direction = {"Age": False,"Salt_Intake": False,"Stress_Score": False,"Sleep_Duration": True,"BMI": True,"Exercise_Level": False}
        name_mapping = {
            "Age":"年龄","Salt_Intake":"盐摄入量","Stress_Score":"压力评分","Sleep_Duration":"睡眠时长","BMI":"BMI",
            "Exercise_Level":"运动水平","Family_History":"高血压家族病史",
            "BP_History_Prehypertension":"既往高血压前期","BP_History_Hypertension":"既往高血压病史","Smoking_Status_Smoker":"吸烟状态"
        }
        bad_desc_set = {"年龄偏大", "睡眠时长不足", "盐摄入量偏高", "盐摄入量偏低",
                        "BMI偏高（超重）", "BMI偏低（偏瘦）", "运动量不足", "压力评分较高"}
        good_desc_set = {"BMI处于正常区间", "睡眠时长充足", "运动水平较高", "运动水平中等",
                         "压力评分较低", "盐摄入量处于正常范围","年龄偏小"}

        for feat_name, shap_val in shap_dict.items():
            if feat_name in ["BP_History_Prehypertension", "BP_History_Hypertension", "Smoking_Status_Smoker"]:
                continue
            feat_cn = name_mapping.get(feat_name, feat_name)
            desc = feat_cn
            if feat_name in ["Age", "Salt_Intake", "Stress_Score", "Sleep_Duration", "BMI", "Exercise_Level"]:
                val = raw_data[feat_name]
                if feat_name == "Age":
                    if val >= threshold["Age"]:
                        desc = "年龄偏大"
                    elif val <= threshold["Age_low"]:
                        desc = "年龄偏小"
                    else:
                        desc = "年龄"
                elif feat_name == "Salt_Intake":
                    if val >= threshold["Salt_high"]:
                        desc = "盐摄入量偏高"
                    elif val < threshold["Salt_low"]:
                        desc = "盐摄入量偏低"
                    else:
                        desc = "盐摄入量处于正常范围"
                elif feat_name == "Stress_Score":
                    desc = "压力评分较高" if val >= threshold["Stress_Score"] else "压力评分较低"
                elif feat_name == "Sleep_Duration":
                    desc = "睡眠时长不足" if val < threshold["Sleep_Duration"] else "睡眠时长充足"
                elif feat_name == "BMI":
                    if val >= threshold["BMI"]:
                        desc = "BMI偏高（超重）"
                    elif val <18.5:
                        desc = "BMI偏低（偏瘦）"
                    else:
                        desc = "BMI处于正常区间"
                elif feat_name == "Exercise_Level":
                    if val <=1:
                        desc = "运动量不足"
                    elif val >=3:
                        desc = "运动水平较高"
                    else:
                        desc = "运动水平中等"

                # 26‑44岁，desc=="年龄"，中性，跳过输出
                if desc == "年龄":
                    continue

                dir_flag = feature_direction[feat_name]
                risk_by_shap = False
                if dir_flag:
                    if shap_val >0.01:
                        risk_by_shap = True
                else:
                    if shap_val < -0.01:
                        risk_by_shap = True

                # SHAP贡献大于阈值才加入报告文案
                if abs(shap_val) >0.01:
                    if desc in bad_desc_set:
                        risk_desc_list.append(desc)
                    elif desc in good_desc_set:
                        protect_desc_list.append(desc)
                    else:
                        if risk_by_shap:
                            risk_desc_list.append(desc)
                        else:
                            protect_desc_list.append(desc)
            elif feat_name == "Family_History":
                val = raw_data[feat_name]
                desc = "存在高血压家族病史" if val == 1 else "无高血压家族病史"
                if abs(shap_val) > 0.01:
                    if val ==1:
                        risk_desc_list.append(desc)
                    else:
                        protect_desc_list.append(desc)

        if bp_history == "Hypertension":
            risk_desc_list.append("既往存在高血压病史")
        elif bp_history == "Prehypertension":
            risk_desc_list.append("既往处于高血压前期")
        else:
            protect_desc_list.append("既往自述血压正常")
        if smoke_status == "Smoker":
            risk_desc_list.append("有吸烟习惯")
        else:
            protect_desc_list.append("无吸烟习惯")

        risk_text = "、".join(risk_desc_list) if risk_desc_list else "无显著正向风险因素"
        protect_text = "、".join(protect_desc_list) if protect_desc_list else "无明显保护因素"

        if risk_result == "低风险":
            advice = "✅建议维持现有健康生活习惯，每年进行一次血压体检。"
        elif risk_result == "中风险":
            advice = "⚠️建议定期监测血压，减少高盐饮食，控制体重，规律运动。"
        else:
            advice = "🚨请尽快前往医院心内科完成相关检查，严格控盐，密切监测血压，遵从临床医生指导。"

        st.subheader("📋 风险筛查报告")
        st.write(f"**预估患病概率：{prob*100:.2f}%**")
        st.write(f"**风险等级：{risk_result}**")
        st.write(f"**主要风险因素：** {risk_text}")
        st.write(f"**保护因素：** {protect_text}")
        st.write(f"**健康建议：** {advice}")
        st.caption("本系统仅为流行病学风险筛查工具，不能作为临床确诊依据。")

        st.subheader("🔍 特征贡献SHAP可视化")
        plot_shap(input_scaled, df_input)

# ================= 多人在线录入 =================
elif menu == "👥多人在线录入分析":
    st.header("👥 多人在线录入批量风险评估")
    if "multi_person_df" not in st.session_state:
        st.session_state.multi_person_df = pd.DataFrame([
            {"姓名":"人员A","Age":42,"Salt_Intake":4.5,"Stress_Score":4,"Sleep_Duration":7.0,"BMI":22.5,"Exercise_Level":2,"Family_History":0,"BP_History":"Normal","Smoking_Status":"Non‑Smoker"},
            {"姓名":"人员B","Age":55,"Salt_Intake":7.2,"Stress_Score":7,"Sleep_Duration":6.0,"BMI":26.1,"Exercise_Level":1,"Family_History":1,"BP_History":"Prehypertension","Smoking_Status":"Smoker"},
        ])

    edited_df = st.data_editor(st.session_state.multi_person_df, num_rows="dynamic", use_container_width=True)
    st.session_state.multi_person_df = edited_df

    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        run_batch = st.button("🚀 执行多员批量风险预测")
    with col_btn2:
        st.download_button(
            label="📥下载模板CSV",
            data=edited_df.to_csv(index=False).encode("utf‑8‑sig"),
            file_name="多人录入模板.csv",
            mime="text/csv"
        )

    if run_batch:
        df_batch_raw = edited_df.copy()
        valid_mask, invalid_rows = check_empty_rows(df_batch_raw)
        if len(invalid_rows) > 0:
            st.warning(f"⚠️空白行索引 {invalid_rows}，该行跳过预测。")
        df_valid_raw = df_batch_raw.loc[valid_mask].copy()
        if df_valid_raw.shape[0] == 0:
            st.error("❌无有效输入数据。")
        else:
            df_model_input = preprocess_data(df_valid_raw.drop(columns=["姓名"]))
            batch_scaled = scaler.transform(df_model_input)
            batch_prob = model.predict_proba(batch_scaled)[:,1]
            risk_labels = [risk_level(p) for p in batch_prob]

            output_df = df_valid_raw.copy()
            output_df["患病预测概率(%)"] = np.round(batch_prob*100,2)
            output_df["风险等级"] = risk_labels

            st.subheader("📊 多员批量预测结果")
            st.dataframe(output_df, use_container_width=True)
            st.download_button(
                label="📥下载预测结果CSV",
                data=output_df.to_csv(index=False).encode("utf‑8‑sig"),
                file_name="多人批量预测结果.csv",
                mime="text/csv"
            )
            st.subheader("🔍全局SHAP特征贡献图")
            plot_shap(batch_scaled, df_model_input)

# ================= CSV上传批量 =================
elif menu == "📂CSV数据集上传分析":
    st.header("📂 上传CSV数据集批量风险评估")
    uploaded_csv = st.file_uploader("上传csv数据集", type="csv")
    if uploaded_csv is not None:
        df_batch = pd.read_csv(uploaded_csv)
        st.subheader("数据预览")
        st.dataframe(df_batch.head())
        st.write(f"原始样本量：{df_batch.shape[0]}")

        if st.button("批量运行模型预测"):
            valid_mask, invalid_rows = check_empty_rows(df_batch)
            if len(invalid_rows) >0:
                st.warning(f"⚠️过滤空样本，数量：{len(invalid_rows)}")
            df_valid = df_batch.loc[valid_mask].copy()
            if df_valid.shape[0]==0:
                st.error("❌CSV无有效数据。")
            else:
                df_batch_encode = preprocess_data(df_valid)
                batch_scaled = scaler.transform(df_batch_encode)
                batch_prob = model.predict_proba(batch_scaled)[:,1]
                risk_labels = [risk_level(p) for p in batch_prob]

                result_df = pd.DataFrame({
                    "患病预测概率(%)": np.round(batch_prob*100,2),
                    "风险等级": risk_labels
                })
                st.subheader("批量预测结果")
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    label="📥下载批量预测结果",
                    data=result_df.to_csv(index=False).encode("utf‑8‑sig"),
                    file_name="csv上传批量预测结果.csv",
                    mime="text/csv"
                )
                st.subheader("🔍全局SHAP特征贡献图")
                plot_shap(batch_scaled, df_batch_encode)