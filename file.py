import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


st.set_page_config(
    page_title="Bank Campaign Sense",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e8e8f0;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #9090b0;
    font-weight: 300;
    margin-top: 6px;
    margin-bottom: 24px;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #f7971e;
    margin-bottom: 10px;
    margin-top: 24px;
}

/* Result card */
.result-yes {
    background: linear-gradient(135deg, #1a472a, #2d6a4f);
    border: 1px solid #52b788;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(82,183,136,0.25);
}
.result-no {
    background: linear-gradient(135deg, #4a1530, #7b2d42);
    border: 1px solid #e05c7a;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(224,92,122,0.25);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}
.result-sub {
    font-size: 0.95rem;
    color: #ccc;
    margin-top: 6px;
}
.prob-bar-wrap {
    background: rgba(255,255,255,0.1);
    border-radius: 8px;
    height: 10px;
    margin: 18px 0 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 8px;
    transition: width 0.6s ease;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffd200;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 30, 0.85);
    border-right: 1px solid rgba(247,151,30,0.15);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #bbb !important;
    font-size: 0.85rem !important;
}
.stButton > button {
    background: linear-gradient(90deg, #f7971e, #ffd200);
    color: #0f0c29;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 14px 40px;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.04em;
    margin-top: 12px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 20px 0;
}
.info-pill {
    display: inline-block;
    background: rgba(247,151,30,0.12);
    border: 1px solid rgba(247,151,30,0.3);
    color: #f7971e;
    border-radius: 20px;
    font-size: 0.78rem;
    padding: 3px 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_artifacts():
    artifacts = {}
    files = {
        'model'   : 'best_model.pkl',
        'scaler'  : 'scaler.pkl',
        'le'      : 'label_encoder.pkl',
        'columns' : 'feature_columns.pkl'
    }
    missing = [v for k, v in files.items() if not os.path.exists(v)]
    if missing:
        return None, missing

    for key, fname in files.items():
        with open(fname, 'rb') as f:
            artifacts[key] = pickle.load(f)
    return artifacts, []

artifacts, missing_files = load_artifacts()


SCALE_NEEDED = ['Logistic Regression', 'KNN', 'Naive Bayes']
NOMINAL_COLS  = ['job','marital','education','contact','month','day_of_week','poutcome','housing','loan']

def predict(client_dict, artifacts):
    df_new = pd.DataFrame([client_dict])
    df_new = pd.get_dummies(df_new, columns=[c for c in NOMINAL_COLS if c in df_new.columns], drop_first=True)
    df_new = df_new.reindex(columns=artifacts['columns'], fill_value=0)
    bool_c = df_new.select_dtypes(include='bool').columns
    df_new[bool_c] = df_new[bool_c].astype(int)

    model_name = type(artifacts['model']).__name__
    needs_scale = any(s.replace(' ','').lower() in model_name.lower() for s in ['logistic','kneighbors','gaussiannb'])
    X = artifacts['scaler'].transform(df_new) if needs_scale else df_new.values

    pred  = artifacts['model'].predict(X)[0]
    proba = artifacts['model'].predict_proba(X)[0]
    label = artifacts['le'].inverse_transform([pred])[0]
    return label, proba



col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<p class="hero-title">🏦 Bank Campaign Sense</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Predict term deposit subscription likelihood from client & campaign data</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    model_type = type(artifacts['model']).__name__ if artifacts else "—"
    st.markdown(f'<div style="text-align:right"><span class="info-pill">Model: {model_type}</span></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)



if missing_files:
    st.error(f"❌ Missing model files: `{'`, `'.join(missing_files)}`")
    st.info("👉 Run your Jupyter Notebook first to train and save the model artifacts, then place them in the same folder as `app.py`.")
    st.stop()



with st.sidebar:
    st.markdown('<p class="section-label">👤 Client Profile</p>', unsafe_allow_html=True)

    age = st.slider("Age", 18, 95, 35)
    job = st.selectbox("Job", [
        'admin.','blue-collar','entrepreneur','housemaid','management',
        'retired','self-employed','services','student','technician','unemployed'
    ])
    marital = st.selectbox("Marital Status", ['married','single','divorced'])
    education = st.selectbox("Education", [
        'basic.4y','basic.6y','basic.9y','high.school',
        'illiterate','professional.course','university.degree'
    ])

    st.markdown('<p class="section-label">💳 Financial Info</p>', unsafe_allow_html=True)
    housing = st.selectbox("Housing Loan?", ['yes','no'])
    loan    = st.selectbox("Personal Loan?", ['yes','no'])

    st.markdown('<p class="section-label">📞 Campaign Details</p>', unsafe_allow_html=True)
    contact     = st.selectbox("Contact Type",    ['cellular','telephone'])
    month       = st.selectbox("Last Contact Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
    day_of_week = st.selectbox("Day of Week",     ['mon','tue','wed','thu','fri'])
    duration    = st.slider("Call Duration (sec)", 0, 4000, 250)
    campaign    = st.slider("Contacts This Campaign", 1, 50, 2)

    st.markdown('<p class="section-label">🔄 Previous Campaign</p>', unsafe_allow_html=True)
    previous        = st.slider("Previous Contacts", 0, 10, 0)
    poutcome        = st.selectbox("Previous Outcome", ['nonexistent','failure','success'])
    prev_contacted  = 1 if poutcome != 'nonexistent' else 0
    pdays           = st.slider("Days Since Last Contact (0 = not prev. contacted)", 0, 30, 0)

    st.markdown('<p class="section-label">📈 Economic Indicators</p>', unsafe_allow_html=True)
    emp_var_rate    = st.number_input("Employment Variation Rate",  -4.0, 2.0,  1.1,  0.1)
    cons_price_idx  = st.number_input("Consumer Price Index",       92.0, 95.0, 93.9, 0.1)
    cons_conf_idx   = st.number_input("Consumer Confidence Index", -55.0, -25.0, -36.4, 0.1)
    euribor3m       = st.number_input("Euribor 3M Rate",            0.5,  6.0,  4.8,  0.1)
    nr_employed     = st.number_input("Nr. Employed (thousands)",  4900, 5300, 5191, 10)

    predict_btn = st.button("🔮 Predict Subscription")



main_col, side_col = st.columns([3, 2], gap="large")

with main_col:
    if predict_btn:
        client = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'housing': housing, 'loan': loan, 'contact': contact,
            'month': month, 'day_of_week': day_of_week, 'duration': duration,
            'campaign': campaign, 'pdays': float(pdays), 'previous': previous,
            'poutcome': poutcome, 'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m, 'nr.employed': float(nr_employed),
            'prev_contacted': prev_contacted
        }

        label, proba = predict(client, artifacts)
        prob_yes = proba[1]
        prob_no  = proba[0]

        if label == 'yes':
            card_class = "result-yes"
            emoji = "✅"
            result_text = "WILL SUBSCRIBE"
            bar_color   = "#52b788"
            advice      = "High potential lead — prioritize for follow-up calls."
        else:
            card_class = "result-no"
            emoji = "❌"
            result_text = "WON'T SUBSCRIBE"
            bar_color   = "#e05c7a"
            advice      = "Low conversion likelihood — consider deprioritizing."

        st.markdown(f"""
        <div class="{card_class}">
            <p class="result-label">{emoji} {result_text}</p>
            <p class="result-sub">{advice}</p>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{prob_yes*100:.1f}%; background:{bar_color};"></div>
            </div>
            <p style="font-size:0.85rem; color:#ccc; margin:0;">
                Subscription Probability: <strong style="color:{bar_color}">{prob_yes*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability breakdown
        st.markdown('<p class="section-label">📊 Probability Breakdown</p>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#52b788">{prob_yes*100:.1f}%</div>
                <div class="metric-lbl">P(Subscribe = Yes)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#e05c7a">{prob_no*100:.1f}%</div>
                <div class="metric-lbl">P(Subscribe = No)</div>
            </div>""", unsafe_allow_html=True)

        # Key input summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">📋 Input Summary</p>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            'Feature': ['Age', 'Job', 'Education', 'Call Duration', 'Previous Outcome', 'Month'],
            'Value'  : [age, job, education, f"{duration}s", poutcome, month]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="
            border: 1.5px dashed rgba(247,151,30,0.25);
            border-radius: 16px;
            padding: 56px 32px;
            text-align: center;
            color: #666;
        ">
            <div style="font-size:3rem; margin-bottom:12px">🎯</div>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#888; margin:0">
                Fill in the client details in the sidebar<br>and click <strong style="color:#f7971e">Predict Subscription</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

with side_col:
    st.markdown('<p class="section-label">📌 How It Works</p>', unsafe_allow_html=True)
    steps = [
        ("1", "Fill client profile", "Age, job, marital status, education"),
        ("2", "Add campaign info", "Contact type, month, call duration"),
        ("3", "Set economic context", "Euribor rate, employment stats"),
        ("4", "Click Predict", "Model returns subscription probability"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:12px; margin-bottom:14px; align-items:flex-start">
            <div style="
                background: rgba(247,151,30,0.15);
                border: 1px solid rgba(247,151,30,0.4);
                border-radius: 50%;
                width:28px; height:28px;
                display:flex; align-items:center; justify-content:center;
                font-family:'Syne',sans-serif; font-weight:700;
                color:#f7971e; font-size:0.85rem; flex-shrink:0;
            ">{num}</div>
            <div>
                <div style="font-weight:500; color:#ddd; font-size:0.9rem">{title}</div>
                <div style="color:#777; font-size:0.8rem; margin-top:2px">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚡ Key Predictors</p>', unsafe_allow_html=True)
    predictors = [
        ("📞", "Call Duration", "Longer calls → higher conversion"),
        ("🔄", "Previous Outcome", "'Success' = strongest signal"),
        ("📅", "Month", "Mar/Oct/Dec best months"),
        ("👴", "Age Group", "Students & retirees convert more"),
    ]
    for icon, name, tip in predictors:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border-left: 3px solid rgba(247,151,30,0.5);
            border-radius: 0 8px 8px 0;
            padding: 10px 14px;
            margin-bottom: 8px;
        ">
            <span style="font-size:1rem">{icon}</span>
            <span style="font-weight:500; color:#ddd; font-size:0.88rem; margin-left:6px">{name}</span>
            <div style="color:#777; font-size:0.78rem; margin-top:3px; margin-left:22px">{tip}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#555; text-align:center">
        Bank Campaign Sense · Data Science Project<br>
        Model trained on 41,188 bank marketing records
    </div>
    """, unsafe_allow_html=True)
