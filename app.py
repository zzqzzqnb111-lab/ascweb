#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cervical Adenosquamous Carcinoma Survival Prediction Tool
Cox Proportional Hazards Model — 3-Year & 5-Year OS
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 页面配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Cervical Adenosquamous Carcinoma Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS 样式 ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a4d8f;
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a4d8f 0%, #2e86de 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .result-label {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.4rem;
    }
    .risk-box {
        padding: 1rem 1.4rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1rem;
    }
    .info-box {
        background: #f0f4fa;
        border-left: 4px solid #1a4d8f;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-top: 1.2rem;
        font-size: 0.92rem;
        color: #444;
    }
    .divider { margin: 1.2rem 0; border-top: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# ── 加载模型 ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    cph           = joblib.load('cox_model.pkl')
    cat_cols_order = joblib.load('cat_cols_order.pkl')
    return cph, cat_cols_order

cph, cat_cols_order = load_model()

# ── 编码函数 ──────────────────────────────────────────────────
def encode_input(age, figo, tumor_size, surgery, lymphad, chemo, cph, cat_cols_order):
    row = {
        'FIGO stage':       figo,
        'Tumor size group': tumor_size,
        'Surgery':          surgery,
        'lymphadenectomy':  lymphad,
        'Chemotherapy':     chemo,
    }
    df_row = pd.DataFrame([row])
    for col, cats in cat_cols_order.items():
        df_row[col] = pd.Categorical(df_row[col], categories=cats)
    dummies = pd.get_dummies(df_row[list(cat_cols_order.keys())], drop_first=True)
    result  = pd.concat([pd.DataFrame({'Age': [age]}),
                         dummies.reset_index(drop=True)], axis=1)
    # 对模型中有但当前行没有的列补0
    for c in cph.params_.index:
        if c not in result.columns:
            result[c] = 0
    result = result[cph.params_.index.tolist()]
    return result

def predict(age, figo, tumor_size, surgery, lymphad, chemo):
    X  = encode_input(age, figo, tumor_size, surgery, lymphad, chemo, cph, cat_cols_order)
    sf = cph.predict_survival_function(X)   # DataFrame, index=time
    s3 = float(sf[sf.index <= 36].iloc[-1].values[0]) if (sf.index <= 36).any() else float(sf.iloc[0].values[0])
    s5 = float(sf[sf.index <= 60].iloc[-1].values[0]) if (sf.index <= 60).any() else float(sf.iloc[0].values[0])
    return np.clip(s3, 0, 1), np.clip(s5, 0, 1)

# ── 风险等级 ──────────────────────────────────────────────────
def risk_level(prob):
    if prob >= 0.80:
        return "Low Risk", "#d4edda", "#155724", "🟢"
    elif prob >= 0.60:
        return "Low-Moderate Risk", "#fff3cd", "#856404", "🟡"
    elif prob >= 0.40:
        return "Moderate-High Risk", "#ffe5cc", "#7d4e00", "🟠"
    else:
        return "High Risk", "#f8d7da", "#721c24", "🔴"

# ── 界面 ──────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Cervical Adenosquamous Carcinoma<br>Survival Prediction Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cox Proportional Hazards Model · 3-Year & 5-Year Overall Survival</div>', unsafe_allow_html=True)

# 侧边栏说明
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
**How to use:**
1. Enter patient clinical information in the form
2. Click **"Calculate Survival Probability"**
3. View predicted 3-year and 5-year OS

**Model:** Cox Proportional Hazards  
**Cohort:** n = 1,646 patients  
**Features:** 6 clinical factors

---
**⚠️ Disclaimer**  
For research reference only.  
Cannot replace clinical judgment.
""")
    st.markdown("---")
    st.markdown("### 📊 Features Used")
    st.markdown("""
- Age
- FIGO Stage
- Tumor Size
- Surgery
- Lymphadenectomy
- Chemotherapy
""")

# 输入表单
st.markdown("### 📝 Patient Clinical Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=18, max_value=100, value=50, step=1,
        help="Patient age at diagnosis"
    )

    figo = st.selectbox(
        "FIGO Stage",
        ['ⅠA','ⅠB','ⅡA','ⅡB','ⅢA','ⅢB','ⅢC','ⅣA','ⅣB'],
        index=1,
        help="FIGO 2018 staging"
    )

    tumor_size = st.selectbox(
        "Tumor Size",
        ['≤2','2-4','＞4','Unknown'],
        help="Tumor diameter (cm)"
    )

with col2:
    surgery = st.selectbox(
        "Surgery",
        ['No','Yes'],
        index=1,
        help="Whether surgical treatment was performed"
    )

    lymphad = st.selectbox(
        "Lymphadenectomy",
        ['No','Yes'],
        index=1,
        help="Whether lymph node dissection was performed"
    )

    chemo = st.selectbox(
        "Chemotherapy",
        ['No/Unknown','Yes'],
        help="Whether chemotherapy was administered"
    )

# 预测按钮
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
predict_btn = st.button("🔮 Calculate Survival Probability", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Running Cox model prediction..."):
        try:
            s3, s5 = predict(age, figo, tumor_size, surgery, lymphad, chemo)

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">3-Year Overall Survival</div>
                    <div class="result-value">{s3*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="result-card" style="background: linear-gradient(135deg, #145a32 0%, #27ae60 100%);">
                    <div class="result-label">5-Year Overall Survival</div>
                    <div class="result-value">{s5*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # 风险评级（基于5年生存率）
            st.markdown("")
            level, bg, fg, icon = risk_level(s5)
            st.markdown(f"""
            <div class="risk-box" style="background:{bg}; color:{fg}; border-left: 4px solid {fg};">
                <strong>{icon} Risk Level (based on 5-year OS): {level}</strong><br>
                {"Good prognosis. Regular surveillance recommended." if s5>=0.8
                 else "Moderate prognosis. Close follow-up recommended." if s5>=0.6
                 else "Unfavorable prognosis. Active treatment and intensive monitoring recommended." if s5>=0.4
                 else "Poor prognosis. Multidisciplinary management and intensive care strongly recommended."}
            </div>
            """, unsafe_allow_html=True)

            # 输入摘要
            st.markdown("---")
            st.markdown("#### Input Summary")
            summary_df = pd.DataFrame({
                'Feature': ['Age', 'FIGO Stage', 'Tumor Size', 'Surgery', 'Lymphadenectomy', 'Chemotherapy'],
                'Value':   [f"{age} years", figo, tumor_size, surgery, lymphad, chemo]
            })
            st.table(summary_df.set_index('Feature'))

            # 免责声明
            st.markdown("""
            <div class="info-box">
                <strong>⚠️ Important Notice:</strong> This tool is based on a Cox Proportional Hazards model trained on 
                1,646 cervical adenosquamous carcinoma patients. Predictions are provided for research reference only 
                and <strong>cannot replace</strong> clinical judgment. Individual prognosis is influenced by many factors 
                not captured here. Please integrate findings with clinical expertise and patient-specific conditions.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check that model files (cox_model.pkl, cat_cols_order.pkl) are in the same directory as app.py")
