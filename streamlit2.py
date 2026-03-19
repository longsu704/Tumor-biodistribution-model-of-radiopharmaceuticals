import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ====================== 页面核心配置 ======================
st.set_page_config(
    page_title="Radiopharmaceutical Prediction Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📊"
)

# 学术期刊级CSS美化（核心优化）
st.markdown("""
<style>
    /* 全局样式 */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif;
        font-size: 14px;
        color: #212529;
        background-color: #f8f9fa;
    }

    /* 标题样式 */
    h1 {
        color: #1a476f;
        font-weight: bold;
        font-size: 28px;
        border-bottom: 3px solid #2c5282;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    h2 {
        color: #2c5282;
        font-weight: 600;
        font-size: 22px;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }

    /* 模型选择卡片 */
    .model-card {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #dee2e6;
        margin-bottom: 15px;
    }

    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .model-card.selected {
        border: 2px solid #2c5282;
        background-color: #e8f4f8;
    }

    /* 表单样式 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px 12px;
    }

    /* 按钮样式 */
    .stButton > button {
        background-color: #2c5282;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1a365d;
        transform: translateY(-1px);
    }

    /* 结果卡片 */
    .result-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #2c5282;
        margin: 15px 0;
    }

    /* 数据表格样式 */
    .dataframe {
        border: 1px solid #dee2e6;
        border-radius: 4px;
    }

    /* 分隔线 */
    .divider {
        border-top: 1px solid #dee2e6;
        margin: 20px 0;
    }

    /* 历史记录样式 */
    .history-item {
        padding: 10px;
        border-bottom: 1px solid #f1f3f5;
    }

    .history-item:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# ====================== 初始化Session State ======================
# 初始化选中的模型
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "pred_model_XGBoost_feat-basic"

# 初始化历史记录（最多保存10条）
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


# ====================== 加载模型函数（保留缓存） ======================
@st.cache_resource
def load_model(prefix):
    try:
        model = joblib.load(f'{prefix}_model.pkl')
        num_imputer = joblib.load(f'{prefix}_num_imputer.pkl')
        scaler = joblib.load(f'{prefix}_scaler.pkl')
        te = joblib.load(f'{prefix}_te.pkl')
        selected_feats = joblib.load(f'{prefix}_selected_feats.pkl')
        num_cols = joblib.load(f'{prefix}_num_cols.pkl')
        feature_names = joblib.load(f'{prefix}_feature_names.pkl')
        cat_unique_vals = joblib.load(f'{prefix}_cat_unique_vals.pkl')
        return model, num_imputer, scaler, te, selected_feats, num_cols, feature_names, cat_unique_vals
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e}. Please run the training scripts first.")
        return None


# ====================== 模型选择（并列卡片式） ======================
def select_model(model_name):
    st.session_state.selected_model = model_name
    st.rerun()  # 关键：强制页面刷新，确保选中状态同步

# ====================== 主界面 ======================
# 标题区域
st.title("Radiopharmaceutical Biodistribution Prediction Platform")
st.markdown("""
*A machine learning web page based on radiopharmaceutical properties to predict tumor biodistribution in mice*  
""")

# 模型选择区域
st.subheader("1. Model Selection")
col_model1, col_model2 = st.columns(2, gap="large")

# 模型1卡片
with col_model1:
    is_selected1 = st.session_state.selected_model == "pred_model_XGBoost_feat-basic"
    card_class = "model-card selected" if is_selected1 else "model-card"
    st.markdown(f"""
    <div class="{card_class}">
        <h3>pred_model_xgboost_feat-basic</h3>
        <p style='color:#6c757d;'>Basic feature set with XGBoost regression</p>
    </div>
    """, unsafe_allow_html=True)
    # 按钮辅助选择
    if st.button("Select This Model", key="btn1"):
        select_model("pred_model_XGBoost_feat-basic")

# 模型2卡片
with col_model2:
    is_selected2 = st.session_state.selected_model == "pred_model_CatBoost_feat-chelator"
    card_class = "model-card selected" if is_selected2 else "model-card"
    st.markdown(f"""
    <div class="{card_class}">
        <h3>pred_model_catboost_feat-chelator</h3>
        <p style='color:#6c757d;'>Chelator-enhanced features with CatBoost regression</p>
    </div>
    """, unsafe_allow_html=True)
    # 按钮辅助选择
    if st.button("Select This Model", key="btn2"):
        select_model("pred_model_CatBoost_feat-chelator")

# 显示当前选中的模型
st.markdown(f"""
<div style='color:#2c5282; font-weight:600; margin:10px 0;'>
    Current Selected Model: {st.session_state.selected_model}
</div>
""", unsafe_allow_html=True)

# 分隔线
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ====================== 加载选中的模型 ======================
prefix = "xgboost" if "xgboost" in st.session_state.selected_model else "catboost"
components = load_model(prefix)

if components:
    model, num_imputer, scaler, te, selected_feats, num_cols, feature_names, cat_unique_vals = components

    # 确定特征数量
    if prefix == "xgboost":
        n_cat = 6
        n_num = 6
    else:
        n_cat = 7
        n_num = 6

    cat_cols = feature_names[:n_cat]
    num_cols_input = feature_names[n_cat: n_cat + n_num]

    # ====================== 输入表单 ======================
    st.subheader("2. Input Parameters")
    input_dict = {}

    # 分栏布局：类别特征 + 数值特征
    col_cat, col_num = st.columns(2, gap="large")

    # 左侧：类别特征（下拉选择，按字母排序）
    with col_cat:
        st.markdown("<h3 style='font-size:16px;'>Categorical Features</h3>", unsafe_allow_html=True)
        for i, col in enumerate(cat_cols):
            options = cat_unique_vals.get(col, [])
            input_dict[col] = st.selectbox(
                label=f"{col}",
                options=options,
                key=f"cat_{i}",
                help=f"Select value for {col} (pre-sorted alphabetically)"
            )

    # 右侧：数值特征（支持NA输入）
    with col_num:
        st.markdown("<h3 style='font-size:16px;'>Numerical Features</h3>", unsafe_allow_html=True)
        for i, col in enumerate(num_cols_input):
            val_str = st.text_input(
                label=f"{col}",
                value="NA",
                key=f"num_{i}",
                help="Enter numerical value or 'NA' for missing data"
            )

            # 处理NA和数值转换
            if val_str.strip().upper() == "NA":
                input_dict[col] = np.nan
            else:
                try:
                    input_dict[col] = float(val_str)
                except ValueError:
                    st.warning(f"Invalid input for {col} - using NA instead")
                    input_dict[col] = np.nan

    # ====================== 预测按钮与逻辑 ======================
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("3. Prediction & Results")

    if st.button("Run Prediction", type="primary"):
        with st.spinner('Processing data (journal-standard validation)...'):
            # 1. 构建输入DataFrame
            input_df = pd.DataFrame([input_dict])

            # 2. 显示用户输入的原始数据
            st.markdown("<h3 style='font-size:16px;'>User Input Data</h3>", unsafe_allow_html=True)
            # 美化输入数据展示
            input_display = input_df.copy()
            input_display = input_display.replace({np.nan: "NA"})  # 替换NaN为NA显示
            st.dataframe(
                input_display.style.set_properties(**{
                    'background-color': 'white',
                    'border': '1px solid #dee2e6',
                    'padding': '8px'
                }),
                use_container_width=True
            )

            # 3. 数据预处理（和原始逻辑一致）
            X_cat = input_df[cat_cols].copy()
            X_num = input_df[num_cols_input].copy()

            # 数值列处理
            for col in X_num.columns:
                X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

            X_num_imputed = pd.DataFrame(num_imputer.transform(X_num), columns=X_num.columns)
            X_num_scaled = pd.DataFrame(scaler.transform(X_num_imputed), columns=X_num.columns)

            # 特征交互
            current_num_cols = num_cols
            for i in range(len(current_num_cols)):
                for j in range(i + 1, min(i + 3, len(current_num_cols))):
                    col1, col2 = current_num_cols[i], current_num_cols[j]
                    if col1 in X_num_scaled.columns and col2 in X_num_scaled.columns:
                        X_num_scaled[f'{col1}_mul_{col2}'] = X_num_scaled[col1] * X_num_scaled[col2]

            for col in current_num_cols[:3]:
                if col in X_num_scaled.columns:
                    X_num_scaled[f'{col}_sq'] = X_num_scaled[col] ** 2
                    X_num_scaled[f'{col}_log'] = np.log1p(np.abs(X_num_scaled[col]))

            if len(current_num_cols) >= 2:
                c1, c2 = current_num_cols[0], current_num_cols[1]
                if c1 in X_num_scaled.columns and c2 in X_num_scaled.columns:
                    X_num_scaled[f'{c1}_div_{c2}'] = X_num_scaled[c1] / (X_num_scaled[c2] + 1e-6)

            # 类别列处理
            X_cat = X_cat.fillna('NA').astype(str)
            X_cat_encoded = te.transform(X_cat)

            # 特征合并与对齐
            X_processed = pd.concat([X_cat_encoded, X_num_scaled], axis=1)
            X_processed = X_processed.fillna(0)

            for feat in selected_feats:
                if feat not in X_processed.columns:
                    X_processed[feat] = 0
            X_final = X_processed[selected_feats]

            # 4. 预测
            prediction = model.predict(X_final)[0]

            # 5. 美化结果展示
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="result-card">
                <h3 style='margin:0;'>Prediction Result</h3>
                <p style='font-size:18px; font-weight:bold; color:#2c5282; margin:10px 0;'>
                    {:.4f}
                </p>
                <p style='color:#6c757d; margin:0;'>
                    Model: {} | Calculation Time: {}
                </p>
            </div>
            """.format(prediction, st.session_state.selected_model, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        unsafe_allow_html=True)

            # 6. 保存到历史记录
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': st.session_state.selected_model,
                'input_data': input_dict,
                'prediction': round(prediction, 4)
            }

            # 添加到历史记录，保持最多10条
            st.session_state.prediction_history.insert(0, history_entry)
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[:10]

    # ====================== 最近10条历史记录======================
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("4. Recent Prediction History (Last 10)")

    if st.session_state.prediction_history:
        # 构建历史记录DataFrame
        history_data = []
        for entry in st.session_state.prediction_history:
            # 处理输入数据展示
            input_str = ", ".join([f"{k}: {v if not pd.isna(v) else 'NA'}" for k, v in entry['input_data'].items()])
            history_data.append({
                'Time': entry['timestamp'],
                'Model': entry['model'],
                'Prediction Value': entry['prediction'],
                'Input Summary': input_str[:100] + "..." if len(input_str) > 100 else input_str
            })

        history_df = pd.DataFrame(history_data)
        # 美化历史记录表格
        st.dataframe(
            history_df.style.set_properties(**{
                'background-color': 'white',
                'border': '1px solid #dee2e6',
                'padding': '8px'
            }),
            use_container_width=True,
            hide_index=True
        )

        # 清空历史记录按钮
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.markdown("<p style='color:#6c757d;'>No prediction history yet</p>", unsafe_allow_html=True)

else:
    st.error("Failed to load model components. Please check model files.")
    st.stop()

# 页脚
st.markdown("""
<div style='margin-top:50px; padding-top:20px; border-top:1px solid #dee2e6; color:#6c757d; text-align:center;'>
    Radiopharmaceutical Biodistribution Prediction Platform | Designed for Academic Standards
</div>
""", unsafe_allow_html=True)
