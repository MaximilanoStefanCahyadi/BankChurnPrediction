import os, pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              average_precision_score, confusion_matrix)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Intelligence", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer { visibility: hidden; }
header { background-color: transparent !important; }

.block-container { padding: 0 2rem 2rem 2rem !important; }

section[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
  border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size:12px !important; }
section[data-testid="stSidebar"] .stSelectbox>div>div {
  background:#1e293b; border:1px solid #334155; color:#e2e8f0;
}

.metric-card {
  background:white; border-radius:14px; padding:1.25rem 1.5rem;
  box-shadow:0 1px 3px rgba(0,0,0,.07),0 4px 16px rgba(0,0,0,.04);
  border:1px solid #f1f5f9; height:100%;
}
.metric-label { font-size:12px; font-weight:500; color:#94a3b8;
  letter-spacing:.05em; text-transform:uppercase; margin-bottom:.4rem; }
.metric-value { font-size:2rem; font-weight:700; line-height:1; color:#0f172a; }
.metric-sub   { font-size:12px; color:#64748b; margin-top:.3rem; }

.section-header {
  font-size:16px; font-weight:600; letter-spacing:.02em;
  color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:.4rem; margin-bottom:1rem; margin-top:1.5rem;
}

.pred-card {
  background:white; border-radius:16px; padding:2rem;
  box-shadow:0 2px 8px rgba(0,0,0,.08); border:1px solid #f1f5f9; text-align:center;
}
.pred-prob  { font-size:4rem; font-weight:800; line-height:1; }
.pred-label { font-size:13px; color:#64748b; margin-top:.25rem; }

.badge { display:inline-block; padding:4px 14px; border-radius:99px;
  font-size:13px; font-weight:600; margin-bottom:1rem; }
.badge-green  { background:#dcfce7; color:#16a34a; }
.badge-yellow { background:#fef9c3; color:#ca8a04; }
.badge-red    { background:#fee2e2; color:#dc2626; }

.conf-track { background:#f1f5f9; border-radius:99px; height:10px;
  overflow:hidden; margin:.5rem 0; }
.conf-fill  { height:100%; border-radius:99px; transition: width 0.4s ease-in-out; }

.split-bar  { display:flex; border-radius:99px; overflow:hidden;
  height:28px; margin:.75rem 0; border:1px solid #e2e8f0; }
.split-stay  { background:#22c55e; display:flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; color:white; transition: width 0.4s; }
.split-churn { background:#ef4444; display:flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; color:white; transition: width 0.4s; }

.factor-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.factor-name { font-size:12px; color:#475569; width:130px; flex-shrink:0; }
.factor-bar-wrap { flex:1; background:#f1f5f9; border-radius:4px; height:8px; }
.factor-bar { height:8px; border-radius:4px; background:#3b82f6; transition: width 0.4s; }
.factor-pct { font-size:11px; color:#94a3b8; width:36px; text-align:right; }

.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#f8fafc; border-radius:10px;
  padding:4px; border:1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
  border-radius:7px; padding:8px 20px; font-size:13px; font-weight:500; color:#64748b;
}
.stTabs [aria-selected="true"] {
  background:white !important; color:#0f172a !important;
  box-shadow:0 1px 3px rgba(0,0,0,.1);
}
.page-title { font-size:28px; font-weight:700; color:#0f172a; margin-bottom:.25rem; }
.page-sub   { font-size:14px; color:#64748b; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'credit_score', 'age', 'tenure', 'balance', 'products_number', 
    'credit_card', 'active_member', 'estimated_salary', 
    'country_Germany', 'country_Spain', 'gender_Male'
]

FEATURE_LABELS = ['Credit score','Age','Tenure','Balance','Products',
                  'Credit card','Active member','Est. salary',
                  'Country: Germany','Country: Spain','Gender: Male']

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Loads both the Baseline and SMOTE models into a dictionary."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}
    
    model_files = {
        "SMOTE Optimized": "smote.pkl",
        "Base Model": "base.pkl"
    }
    
    for m_name, f_name in model_files.items():
        for p in [os.path.join(base_dir, f_name), f_name]:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    models[m_name] = pickle.load(f)
                break
    return models

@st.cache_resource
def load_scaler():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'scaler.pkl'), 'scaler.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("scaler.pkl not found — place it next to app.py")

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'churn.csv'), 'churn.csv']:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'customer_id' in df.columns:
                df = df.drop(columns=['customer_id'])
            return df
    raise FileNotFoundError("churn.csv not found")

@st.cache_data
def prepare_test_set(_scaler):
    df = load_data()
    df_enc = pd.get_dummies(df, columns=['country','gender'], drop_first=True)
    
    for col in FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = 0
            
    X = df_enc[FEATURES].astype(float)
    y = df_enc['churn']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_test_scaled = X_test.copy()
    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary']
    X_test_scaled[cols_to_scale] = _scaler.transform(X_test[cols_to_scale])
    
    return X_test_scaled, y_test

# Load everything globally
models_dict = load_models()
scaler = load_scaler()
df = load_data()
X_test_scaled, y_test = prepare_test_set(scaler)

def build_input_df(credit_score, age, tenure, balance, products,
                   credit_card, active, country, gender):
    row = {
        'credit_score'     : float(credit_score),
        'age'              : float(age),
        'tenure'           : float(tenure),
        'balance'          : float(balance),
        'products_number'  : float(products),
        'credit_card'      : float(credit_card),
        'active_member'    : float(active),
        'estimated_salary' : float(0),
        'country_Germany'  : float(country == 'Germany'),
        'country_Spain'    : float(country == 'Spain'),
        'gender_Male'      : float(gender == 'Male'),
    }
    return pd.DataFrame([row])[FEATURES]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.25rem 0 1rem">
      <div style="font-size:20px;font-weight:700;color:#f8fafc">🏦 Churn Intelligence</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px">Gradient Boosting Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:.5rem;">⚙️ Model Selection</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active Prediction Engine:", list(models_dict.keys()))
    
    active_model = models_dict[selected_model_name]

    st.markdown("<hr style='border-color:#334155;margin:1rem 0 .75rem 0'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:11px;font-weight:600;letter-spacing:.08em;
         text-transform:uppercase;color:#64748b;margin-bottom:.75rem">
      Customer Profile
    </div>""", unsafe_allow_html=True)

    country  = st.selectbox("Country",  ['France', 'Germany', 'Spain'])
    gender   = st.selectbox("Gender",   ['Female', 'Male'])
    age      = st.slider("Age",          18, 92,  39)
    credit_score = st.slider("Credit score", 350, 850, 650)
    tenure   = st.slider("Tenure (years)", 0, 10, 5)
    balance  = st.slider("Balance ($)", 0, 250000, 76000, step=500)
    salary   = st.slider("Estimated salary ($)", 0, 200000, 100000, step=500)

    st.markdown("<hr style='border-color:#334155;margin:.75rem 0'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:11px;font-weight:600;letter-spacing:.08em;
        text-transform:uppercase;color:#64748b;margin-bottom:.75rem">Behaviour</div>""",
        unsafe_allow_html=True)

    products    = st.select_slider("Number of products", [1, 2, 3, 4], value=1)
    credit_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    active      = st.radio("Active member?",   ["Yes", "No"], horizontal=True)

    cc_val = 1 if credit_card == "Yes" else 0
    am_val = 1 if active      == "Yes" else 0

    input_df = build_input_df(credit_score, age, tenure, balance,
                              products, cc_val, am_val, country, gender)
    input_df['estimated_salary'] = float(salary)

    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary']
    input_scaled = input_df.copy()
    input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    churn_prob = float(active_model.predict_proba(input_scaled)[0][1])
    stay_prob  = 1 - churn_prob
    confidence = abs(churn_prob - 0.5) * 2

    if churn_prob < 0.30:
        verdict, badge_cls, bar_color = "Not Churning", "badge-green",  "#22c55e"
    elif churn_prob < 0.55:
        verdict, badge_cls, bar_color = "At Risk",      "badge-yellow", "#f59e0b"
    else:
        verdict, badge_cls, bar_color = "Likely Churning", "badge-red", "#ef4444"

    conf_label = "High"   if confidence >= 0.7 else \
                 "Medium" if confidence >= 0.4 else "Low"
    conf_color = "#22c55e" if confidence >= 0.7 else \
                 "#f59e0b" if confidence >= 0.4 else "#ef4444"


# Get test predictions for the currently active model (for Tab 1 metrics)
proba_test_active = active_model.predict_proba(X_test_scaled)[:,1]


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">Bank Customer Churn Intelligence</div>
<div class="page-sub">Predict churn risk and explore Exploratory Data Analysis (EDA) across 10,000 customers.</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "📊 Exploratory Data Analysis (Notebook)", "📋 Data Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_detail = st.columns([1, 1.6], gap="large")

    with col_pred:
        stay_pct  = round(stay_prob  * 100)
        churn_pct = round(churn_prob * 100)

        st.markdown(f"""
        <div class="pred-card">
          <span class="badge {badge_cls}">{verdict}</span>
          <div class="pred-prob" style="color:{bar_color}">{churn_pct}%</div>
          <div class="pred-label">churn probability</div>
          <div style="margin-top:1.5rem;">
            <div style="display:flex;justify-content:space-between;
                 font-size:12px;color:#94a3b8;margin-bottom:4px;">
              <span>Low risk</span><span>High risk</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{churn_pct}%;background:{bar_color};"></div>
            </div>
          </div>
          <div class="split-bar" style="margin-top:1rem;">
            <div class="split-stay"  style="width:{stay_pct}%;">{stay_pct}% stay</div>
            <div class="split-churn" style="width:{churn_pct}%;">{churn_pct}% churn</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Model Confidence</div>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <div class="metric-value" style="color:{conf_color}">{round(confidence*100)}%</div>
            <span style="font-size:13px;color:#64748b;font-weight:500">{conf_label}</span>
          </div>
          <div class="conf-track" style="margin-top:.5rem;">
            <div class="conf-fill" style="width:{round(confidence*100)}%;background:{conf_color};"></div>
          </div>
          <div class="metric-sub">
            {"High confidence — prediction is reliable."       if conf_label=="High"   else
             "Moderate confidence — review manually."          if conf_label=="Medium" else
             "Low confidence — near decision boundary."}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        k1, k2, k3 = st.columns(3)
        for col, label, val, color in [
            (k1, "Stay Probability",  f"{stay_pct}%",  "#22c55e"),
            (k2, "Churn Probability", f"{churn_pct}%", "#ef4444"),
            (k3, "Confidence",        conf_label,       conf_color),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="color:{color};font-size:1.5rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="font-size:13px;margin-top:1rem;margin-bottom:1rem;">Customer Summary</div>', unsafe_allow_html=True)
        fields = [
            ("Country", country), ("Gender", gender),
            ("Age", age),         ("Credit score", credit_score),
            ("Tenure", f"{tenure} yrs"), ("Balance", f"${balance:,}"),
            ("Products", products), ("Est. salary", f"${salary:,}"),
            ("Credit card", credit_card), ("Active member", active),
        ]
        sc1, sc2 = st.columns(2)
        for i, (k, v) in enumerate(fields):
            with (sc1 if i % 2 == 0 else sc2):
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                     padding:.4rem 0;border-bottom:1px solid #f1f5f9;">
                  <span style="font-size:12px;color:#94a3b8">{k}</span>
                  <span style="font-size:13px;font-weight:500;color:#0f172a">{v}</span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics (Notebook EDA Replication)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    
    # --- ROW 1: Churn Dist & Products ---
    r1c1, r1c2 = st.columns(2)
    
    with r1c1:
        st.markdown('<div class="section-header" style="margin-top:0">1. Churn Distribution</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x='churn', palette='Set2', ax=ax1)
        ax1.set_xlabel('Churn (0 = Stay, 1 = Leave)', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig1)

    with r1c2:
        st.markdown('<div class="section-header" style="margin-top:0">2. Churn Rate by Products</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        product_churn = df.groupby('products_number')['churn'].mean().reset_index()
        sns.barplot(data=product_churn, x='products_number', y='churn', palette='magma', ax=ax4)
        for p in ax4.patches:
            ax4.annotate(f'{p.get_height():.1%}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 4), textcoords='offset points')
        ax4.set_xlabel('Number of Products', fontsize=10)
        ax4.set_ylabel('Churn Rate', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4)

    # --- ROW 2: Demographics & Numerical Distributions ---
    r2c1, r2c2 = st.columns(2)
    
    with r2c1:
        st.markdown('<div class="section-header">3. Churn by Demographics</div>', unsafe_allow_html=True)
        fig3, axes3 = plt.subplots(1, 2, figsize=(7, 4.5))
        sns.countplot(data=df, x='country', hue='churn', palette='Set1', ax=axes3[0])
        axes3[0].set_title('By Country', fontsize=11)
        axes3[0].set_xlabel('')
        
        sns.countplot(data=df, x='gender', hue='churn', palette='Set1', ax=axes3[1])
        axes3[1].set_title('By Gender', fontsize=11)
        axes3[1].set_xlabel('')
        plt.tight_layout()
        st.pyplot(fig3)

    with r2c2:
        st.markdown('<div class="section-header">4. Numerical Feature Distributions</div>', unsafe_allow_html=True)
        num_cols = ['credit_score', 'age', 'balance', 'estimated_salary']
        fig2, axes2 = plt.subplots(2, 2, figsize=(7, 4.5))
        for i, col in enumerate(num_cols):
            row_idx, col_idx = i // 2, i % 2
            sns.histplot(data=df, x=col, kde=True, ax=axes2[row_idx, col_idx], color='#3b82f6', bins=20)
            axes2[row_idx, col_idx].set_ylabel('')
            axes2[row_idx, col_idx].set_xlabel(col.replace('_', ' ').title(), fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

    # --- ROW 3: Correlation Heatmap & Feature Importance ---
    r3c1, r3c2 = st.columns(2)
    
    with r3c1:
        st.markdown('<div class="section-header">5. Correlation Heatmap</div>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        df_encoded = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax5, annot_kws={"size": 7})
        ax5.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig5)

    with r3c2:
        st.markdown(f'<div class="section-header">6. Feature Importance ({selected_model_name})</div>', unsafe_allow_html=True)
        fig8, ax8 = plt.subplots(figsize=(6, 5))
        fi = pd.Series(active_model.feature_importances_, index=FEATURE_LABELS).sort_values(ascending=False)
        sns.barplot(x=fi.values, y=fi.index, palette='magma', ax=ax8)
        ax8.set_xlabel('Importance Score', fontsize=10)
        ax8.set_ylabel('')
        ax8.tick_params(labelsize=10)
        plt.tight_layout()
        st.pyplot(fig8)

    # --- ROW 4: ROC and PR Comparison ---
    st.markdown('<div class="section-header">7. Model Performance Comparison</div>', unsafe_allow_html=True)
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 4.5))
    line_colors = {'SMOTE Optimized': 'darkorange', 'Base Model': '#3b82f6'}
    
    for m_name, m_obj in models_dict.items():
        m_proba = m_obj.predict_proba(X_test_scaled)[:,1]
        m_auc = roc_auc_score(y_test, m_proba)
        m_ap  = average_precision_score(y_test, m_proba)
        fpr, tpr, _ = roc_curve(y_test, m_proba)
        prec, rec, _ = precision_recall_curve(y_test, m_proba)
        
        lw = 2.5 if m_name == selected_model_name else 1.5
        alpha = 1.0 if m_name == selected_model_name else 0.5
        color = line_colors.get(m_name, 'purple')
        
        axes6[0].plot(fpr, tpr, color=color, lw=lw, alpha=alpha, label=f'{m_name} (AUC = {m_auc:.3f})')
        axes6[1].plot(rec, prec, color=color, lw=lw, alpha=alpha, label=f'{m_name} (AP = {m_ap:.3f})')
        
    axes6[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes6[0].set_xlabel('False Positive Rate', fontsize=10)
    axes6[0].set_ylabel('True Positive Rate', fontsize=10)
    axes6[0].set_title('ROC Curve Comparison', fontsize=11)
    axes6[0].legend(loc="lower right", fontsize=9)

    axes6[1].set_xlabel('Recall', fontsize=10)
    axes6[1].set_ylabel('Precision', fontsize=10)
    axes6[1].set_title('Precision-Recall Curve Comparison', fontsize=11)
    axes6[1].legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig6)

    # --- ROW 5: Confusion Matrix Comparison ---
    st.markdown('<div class="section-header">8. Confusion Matrix Comparison</div>', unsafe_allow_html=True)
    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, (m_name, m_obj) in enumerate(models_dict.items()):
        # Get raw 0/1 predictions for the confusion matrix
        m_preds = m_obj.predict(X_test_scaled)
        cm = confusion_matrix(y_test, m_preds)

        # Style the active model to stand out
        is_active = (m_name == selected_model_name)
        cmap = 'Oranges' if m_name == 'SMOTE Optimized' else 'Blues'
        title_weight = 'bold' if is_active else 'normal'

        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes7[i],
                    xticklabels=['Predicted Stay', 'Predicted Churn'], 
                    yticklabels=['Actual Stay', 'Actual Churn'],
                    linewidths=1, linecolor='white')
        
        axes7[i].set_title(f'{m_name}', fontweight=title_weight, fontsize=12, pad=10)

    plt.tight_layout()
    st.pyplot(fig7)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Full Dataset — 10,000 Customers</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        filter_churn   = st.selectbox("Churn status", ["All", "Churned", "Stayed"])
    with fc2:
        filter_country = st.selectbox("Country", ["All"] + sorted(df['country'].unique()))
    with fc3:
        age_range = st.slider("Age range", 18, 92, (18, 92))

    dff = df.copy()
    if filter_churn   == "Churned": dff = dff[dff['churn'] == 1]
    if filter_churn   == "Stayed":  dff = dff[dff['churn'] == 0]
    if filter_country != "All":     dff = dff[dff['country'] == filter_country]
    dff = dff[(dff['age'] >= age_range[0]) & (dff['age'] <= age_range[1])]

    st.markdown(f"""
    <div style="font-size:13px;color:#64748b;margin-bottom:.75rem">
      Showing <b style="color:#0f172a">{len(dff):,}</b> of <b style="color:#0f172a">{len(df):,}</b> customers
    </div>""", unsafe_allow_html=True)

    disp = dff.reset_index(drop=True).copy()
    disp['churn']            = disp['churn'].map({0:'✅ Stay', 1:'🔴 Churn'})
    disp['balance']          = disp['balance'].apply(lambda x: f"${x:,.0f}")
    disp['estimated_salary'] = disp['estimated_salary'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(disp, width='stretch', height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0">Summary Statistics (filtered)</div>', unsafe_allow_html=True)
    desc = dff[['credit_score','age','tenure','balance','estimated_salary']].describe().round(1).T[['mean','std','min','50%','max']]
    desc.columns = ['Mean','Std','Min','Median','Max']
    st.dataframe(desc, width='stretch')