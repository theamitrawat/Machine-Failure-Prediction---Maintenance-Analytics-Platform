# app/streamlit_app.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_machine_status

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_machine_failure_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "machine_failure_prediction_dataset.csv")

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#111827"
CARD     = "#1F2937"
CARD2    = "#374151"
BORDER   = "#374151"
CYAN     = "#06B6D4"
CYAN_LT  = "#67E8F9"
CYAN_DK  = "#0891B2"
TEXT     = "#F9FAFB"
MUTED    = "#9CA3AF"
SUCCESS  = "#10B981"
WARNING  = "#F59E0B"
ORANGE   = "#F97316"
DANGER   = "#EF4444"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family="Inter, sans-serif", size=13),
    margin=dict(t=30, b=20, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=12)),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MachineGuard — Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}}

/* ── App background ── */
.stApp {{
    background-color: {BG};
}}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER};
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    padding: 0;
}}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ── Sidebar brand header ── */
.sb-brand {{
    background: linear-gradient(135deg, {CYAN_DK} 0%, #164E63 100%);
    padding: 24px 20px 20px 20px;
    margin-bottom: 8px;
}}
.sb-brand h1 {{
    color: {TEXT};
    font-size: 1.15rem;
    font-weight: 800;
    margin: 8px 0 2px 0;
    letter-spacing: -0.01em;
}}
.sb-brand p {{
    color: {CYAN_LT};
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0;
}}

/* ── Nav items ── */
[data-testid="stRadio"] label {{
    color: {MUTED} !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 6px 0 !important;
    transition: color 0.15s;
}}
[data-testid="stRadio"] label:hover {{ color: {TEXT} !important; }}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: 3px solid {CYAN};
    border-radius: 12px;
    padding: 18px 20px 14px 20px;
    transition: border-color 0.2s;
}}
[data-testid="metric-container"]:hover {{
    border-top-color: {CYAN_LT};
}}
[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
}}

/* ── Section headers ── */
.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.01em;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER};
    margin-left: 8px;
}}

/* ── Page title ── */
.page-header {{
    padding: 28px 0 8px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 28px;
}}
.page-header h1 {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {TEXT};
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}}
.page-header p {{
    color: {MUTED};
    font-size: 0.92rem;
    margin: 0;
}}

/* ── Chart card wrapper ── */
.chart-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 4px;
}}

/* ── Form ── */
[data-testid="stForm"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 24px !important;
}}
[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, {CYAN_DK}, {CYAN});
    color: {BG};
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.4rem;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(6,182,212,0.3);
}}
[data-testid="stFormSubmitButton"] > button:hover {{
    box-shadow: 0 6px 20px rgba(6,182,212,0.45);
    transform: translateY(-1px);
}}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 7px !important;
    color: {TEXT} !important;
}}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {{
    color: {MUTED} !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

/* ── Result metric row ── */
.result-metric {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}}
.result-metric .rm-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {MUTED};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.result-metric .rm-value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {TEXT};
}}

/* ── Risk badge ── */
.risk-badge {{
    border-radius: 10px;
    padding: 16px 28px;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid;
    margin: 16px 0;
}}

/* ── Recommendation card ── */
.rec-item {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: {CARD2};
    border: 1px solid {BORDER};
    border-left: 4px solid {CYAN};
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: {TEXT};
    line-height: 1.5;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 24px 0 !important; }}

/* ── Sidebar social ── */
.sb-social {{
    margin-top: auto;
    padding: 16px 20px 20px 20px;
    border-top: 1px solid {BORDER};
}}
.sb-social .sb-social-label {{
    font-size: 0.68rem;
    color: {MUTED};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 600;
}}
.sb-social a {{
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 7px 10px;
    border-radius: 7px;
    margin-bottom: 4px;
    transition: background 0.15s;
    color: {TEXT};
}}
.sb-social a:hover {{ background: {CARD2}; }}
.sb-social .li-link {{ color: #38BDF8; }}

/* ── Tag pills ── */
.tag {{
    display: inline-block;
    background: rgba(6,182,212,0.12);
    color: {CYAN_LT};
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
}}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEAT_PATH)
    return model, feature_columns

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model artifacts: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div class="sb-brand">
    <div style="font-size:2rem;margin-bottom:6px;">⚙️</div>
    <h1>MachineGuard</h1>
    <p>Predictive Maintenance AI</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠  Dashboard",
     "🔮  Predict Failure",
     "📊  Data Analytics",
     "ℹ️  About"],
    label_visibility="collapsed",
)

st.sidebar.markdown(f"""
<div class="sb-social">
    <div class="sb-social-label">Connect with me</div>
    <a class="li-link" href="https://www.linkedin.com/in/theamitrawat/" target="_blank">
        <img src="https://img.icons8.com/color/18/linkedin.png"/>
        LinkedIn — Amit Rawat
    </a>
    <a href="https://github.com/theamitrawat" target="_blank"
       style="color:{MUTED};">
        <img src="https://img.icons8.com/ios-glyphs/18/9CA3AF/github.png"/>
        GitHub — theamitrawat
    </a>
</div>
""", unsafe_allow_html=True)

# ── Helper: chart card wrapper ────────────────────────────────────────────────
def chart_card(title, fig, key=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown(f"""
    <div class="page-header">
        <h1>⚙️ Machine Failure Dashboard</h1>
        <p>Real-time overview of machine health across your fleet</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_data()
        total     = len(df)
        failures  = int(df['Machine failure'].sum())
        healthy   = total - failures
        fail_rate = round(failures / total * 100, 2)
        health_rt = round(healthy / total * 100, 2)

        # ── KPI row ──
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Records",     f"{total:,}")
        k2.metric("Healthy Machines",  f"{healthy:,}",  delta=f"{health_rt}%")
        k3.metric("Failures Detected", f"{failures:,}", delta=f"{fail_rate}%", delta_color="inverse")
        k4.metric("Failure Rate",      f"{fail_rate}%")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Row 1: pie + bar ──
        c1, c2 = st.columns(2)

        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=["Healthy", "Failure"],
                values=[healthy, failures],
                hole=0.6,
                marker=dict(colors=[SUCCESS, DANGER],
                            line=dict(color=CARD, width=2)),
                textfont=dict(color=TEXT, size=13),
            ))
            fig_pie.add_annotation(
                text=f"<b>{health_rt}%</b><br><span style='font-size:11px'>Healthy</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color=TEXT),
            )
            fig_pie.update_layout(**PLOTLY_BASE, title=None, showlegend=True)
            chart_card("🟢 Fleet Health Distribution", fig_pie, "pie")

        with c2:
            type_fail = df.groupby('Type')['Machine failure'].mean().reset_index()
            type_fail.columns = ['Type', 'Failure Rate']
            type_fail['Failure Rate'] = (type_fail['Failure Rate'] * 100).round(2)
            type_fail['Color'] = type_fail['Failure Rate'].apply(
                lambda x: DANGER if x > 5 else WARNING if x > 2 else SUCCESS
            )
            fig_bar = go.Figure(go.Bar(
                x=type_fail['Type'],
                y=type_fail['Failure Rate'],
                marker=dict(color=type_fail['Color'],
                            line=dict(color=CARD, width=1)),
                text=type_fail['Failure Rate'].apply(lambda x: f"{x}%"),
                textposition='outside',
                textfont=dict(color=TEXT, size=12),
            ))
            fig_bar.update_layout(**PLOTLY_BASE,
                                  yaxis_title="Failure Rate (%)",
                                  xaxis_title="Machine Type")
            chart_card("📊 Failure Rate by Machine Type", fig_bar, "bar_type")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Row 2: sub-failure breakdown ──
        sub_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        sub_labels = {
            'TWF': 'Tool Wear', 'HDF': 'Heat Dissipation',
            'PWF': 'Power', 'OSF': 'Overstrain', 'RNF': 'Random',
        }
        available_sub = [c for c in sub_cols if c in df.columns]
        if available_sub:
            counts = [int(df[c].sum()) for c in available_sub]
            labels = [sub_labels[c] for c in available_sub]
            colors = [CYAN, CYAN_DK, WARNING, ORANGE, DANGER]
            fig_sub = go.Figure(go.Bar(
                x=labels, y=counts,
                marker=dict(color=colors, line=dict(color=CARD, width=1)),
                text=counts, textposition='outside',
                textfont=dict(color=TEXT, size=12),
            ))
            fig_sub.update_layout(**PLOTLY_BASE,
                                  yaxis_title="Count",
                                  xaxis_title="Failure Mode")
            chart_card("🔴 Failure Count by Mode", fig_sub, "sub_fail")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Row 3: sensor histogram ──
        sensor_cols = [
            'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        ]
        available = [c for c in sensor_cols if c in df.columns]
        if available:
            sel = st.selectbox("Select sensor to inspect", available,
                               key="dash_sensor")
            fig_hist = go.Figure()
            for val, label, color in [(0, 'Healthy', SUCCESS), (1, 'Failure', DANGER)]:
                subset = df[df['Machine failure'] == val][sel]
                fig_hist.add_trace(go.Histogram(
                    x=subset, name=label,
                    marker_color=color,
                    opacity=0.75,
                    nbinsx=50,
                ))
            fig_hist.update_layout(**PLOTLY_BASE,
                                   barmode='overlay',
                                   xaxis_title=sel,
                                   yaxis_title="Count")
            chart_card(f"📈 {sel} Distribution", fig_hist, "hist")

    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT FAILURE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Failure":
    st.markdown(f"""
    <div class="page-header">
        <h1>🔮 Predict Machine Failure</h1>
        <p>Enter live sensor readings to get an instant health assessment and maintenance plan</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown(f"<div class='section-title'>⚙️ Sensor Inputs</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
                                        help="L = Low, M = Medium, H = High quality variant")
            type_map     = {"H": 0, "L": 1, "M": 2}
            type_encoded = type_map[machine_type]
            air_temp = st.number_input("Air Temperature (K)",
                                       min_value=290.0, max_value=320.0,
                                       value=300.0, step=0.1,
                                       help="Typical: 295–305 K")
        with col2:
            proc_temp = st.number_input("Process Temperature (K)",
                                        min_value=300.0, max_value=320.0,
                                        value=310.0, step=0.1,
                                        help="Typical: 305–315 K")
            rpm = st.number_input("Rotational Speed (rpm)",
                                  min_value=1000, max_value=3000,
                                  value=1500, step=10,
                                  help="Typical: 1200–2800 rpm")
        with col3:
            torque = st.number_input("Torque (Nm)",
                                     min_value=0.0, max_value=100.0,
                                     value=40.0, step=0.5,
                                     help="Typical: 3–77 Nm")
            tool_wear = st.number_input("Tool Wear (min)",
                                        min_value=0, max_value=300,
                                        value=100, step=1,
                                        help="Typical: 0–253 min")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🚨 Active Failure Flags</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{MUTED};font-size:0.82rem;margin-top:-10px;margin-bottom:12px;'>Set to 1 only if a sensor alert is currently active on the machine.</p>", unsafe_allow_html=True)

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        twf = fc1.selectbox("TWF", [0, 1], help="Tool Wear Failure")
        hdf = fc2.selectbox("HDF", [0, 1], help="Heat Dissipation Failure")
        pwf = fc3.selectbox("PWF", [0, 1], help="Power Failure")
        osf = fc4.selectbox("OSF", [0, 1], help="Overstrain Failure")
        rnf = fc5.selectbox("RNF", [0, 1], help="Random Failure")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Run Health Assessment", use_container_width=True)

    if submitted:
        input_data = [type_encoded, air_temp, proc_temp, rpm, torque, tool_wear,
                      twf, hdf, pwf, osf, rnf]
        try:
            result = predict_machine_status(model, input_data, feature_columns)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>📋 Assessment Results</div>", unsafe_allow_html=True)

        # ── Result metrics ──
        r1, r2, r3, r4 = st.columns(4)
        status_icon  = "🔴" if result["Prediction"] == "Failure" else "🟢"
        status_color = DANGER if result["Prediction"] == "Failure" else SUCCESS

        r1.markdown(f"""
        <div class="result-metric">
            <div class="rm-label">Machine Status</div>
            <div class="rm-value" style="color:{status_color};">{status_icon} {result['Prediction']}</div>
        </div>""", unsafe_allow_html=True)

        prob_color = DANGER if result['Failure Probability'] > 50 else WARNING if result['Failure Probability'] > 20 else SUCCESS
        r2.markdown(f"""
        <div class="result-metric">
            <div class="rm-label">Failure Probability</div>
            <div class="rm-value" style="color:{prob_color};">{result['Failure Probability']}%</div>
        </div>""", unsafe_allow_html=True)

        hs = result['Health Score']
        hs_color = SUCCESS if hs > 80 else WARNING if hs > 60 else ORANGE if hs > 40 else DANGER
        r3.markdown(f"""
        <div class="result-metric">
            <div class="rm-label">Health Score</div>
            <div class="rm-value" style="color:{hs_color};">{hs}<span style="font-size:1rem;font-weight:400;color:{MUTED};"> / 100</span></div>
        </div>""", unsafe_allow_html=True)

        days = result['Maintenance Due (days)']
        days_color = DANGER if days <= 1 else ORANGE if days <= 7 else WARNING if days <= 14 else SUCCESS
        r4.markdown(f"""
        <div class="result-metric">
            <div class="rm-label">Maintenance Due</div>
            <div class="rm-value" style="color:{days_color};">≤ {days}d</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk badge ──
        risk = result["Risk"]
        risk_cfg = {
            "Low Risk":      (SUCCESS, "rgba(16,185,129,0.1)",  "rgba(16,185,129,0.3)"),
            "Medium Risk":   (WARNING, "rgba(245,158,11,0.1)",  "rgba(245,158,11,0.3)"),
            "High Risk":     (ORANGE,  "rgba(249,115,22,0.1)",  "rgba(249,115,22,0.3)"),
            "Critical Risk": (DANGER,  "rgba(239,68,68,0.1)",   "rgba(239,68,68,0.3)"),
        }
        fg, bg, border = risk_cfg.get(risk, (MUTED, CARD, BORDER))
        st.markdown(f"""
        <div class="risk-badge" style="color:{fg};background:{bg};border-color:{border};">
            ⚠️ &nbsp; Risk Level: {risk}
        </div>""", unsafe_allow_html=True)

        # ── Gauge + Recommendations ──
        g_col, r_col = st.columns([1, 1])

        with g_col:
            st.markdown(f"<div class='section-title'>💓 Health Score Gauge</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hs,
                number={"suffix": "%", "font": {"color": TEXT, "size": 40, "family": "Inter"}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": MUTED,
                        "tickfont": {"color": MUTED, "size": 11},
                        "nticks": 6,
                    },
                    "bar": {"color": hs_color, "thickness": 0.22},
                    "bgcolor": CARD2,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "rgba(239,68,68,0.15)"},
                        {"range": [40, 60], "color": "rgba(249,115,22,0.12)"},
                        {"range": [60, 80], "color": "rgba(245,158,11,0.12)"},
                        {"range": [80, 100], "color": "rgba(16,185,129,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": hs_color, "width": 3},
                        "thickness": 0.85,
                        "value": hs,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT, family="Inter"),
                margin=dict(t=20, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r_col:
            st.markdown(f"<div class='section-title'>🔧 Maintenance Recommendations</div>", unsafe_allow_html=True)
            for rec in result["Recommendations"]:
                st.markdown(f"<div class='rec-item'>{rec}</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Analytics":
    st.markdown(f"""
    <div class="page-header">
        <h1>📊 Data Analytics</h1>
        <p>Explore the dataset, sensor distributions, and feature correlations</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    # ── Dataset preview ──
    st.markdown(f"<div class='section-title'>🗂️ Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Stats ──
    st.markdown(f"<div class='section-title'>📐 Descriptive Statistics</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Correlation heatmap ──
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, DANGER], [0.5, CARD2], [1, CYAN]],
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=10, color=TEXT),
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        **{k: v for k, v in PLOTLY_BASE.items() if k not in ('xaxis', 'yaxis')},
        xaxis=dict(tickfont=dict(color=MUTED, size=10), side="bottom"),
        yaxis=dict(tickfont=dict(color=MUTED, size=10), autorange="reversed"),
        height=480,
    )
    chart_card("🔥 Feature Correlation Heatmap", fig_heat, "heatmap")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Scatter ──
    st.markdown(f"<div class='section-title'>🔵 Sensor Scatter Plot</div>", unsafe_allow_html=True)
    sensor_cols = [c for c in df.columns if c not in ['UDI', 'Product ID', 'Type', 'Machine failure']]
    sc1, sc2 = st.columns(2)
    x_axis = sc1.selectbox("X axis", sensor_cols, index=0, key="sx")
    y_axis = sc2.selectbox("Y axis", sensor_cols, index=1, key="sy")

    fig_scatter = go.Figure()
    for val, label, color in [(0, 'Healthy', SUCCESS), (1, 'Failure', DANGER)]:
        sub = df[df['Machine failure'] == val]
        fig_scatter.add_trace(go.Scatter(
            x=sub[x_axis], y=sub[y_axis],
            mode='markers',
            name=label,
            marker=dict(color=color, size=4, opacity=0.55,
                        line=dict(width=0)),
        ))
    fig_scatter.update_layout(**PLOTLY_BASE,
                              xaxis_title=x_axis,
                              yaxis_title=y_axis)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown(f"""
    <div class="page-header">
        <h1>ℹ️ About MachineGuard</h1>
        <p>Predictive maintenance powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([2, 1])

    with a1:
        st.markdown(f"""
        <div style="color:{TEXT};line-height:1.9;">

        <h3 style="color:{CYAN_LT};font-size:1rem;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:12px;">
            What is MachineGuard?
        </h3>
        <p style="color:{MUTED};font-size:0.92rem;">
            MachineGuard is an end-to-end predictive maintenance platform that uses a trained
            <strong style="color:{TEXT};">XGBoost</strong> classifier to assess machine health
            from real-time sensor readings. It predicts failure probability, generates a health
            score, and provides actionable maintenance recommendations — all in one dashboard.
        </p>

        <h3 style="color:{CYAN_LT};font-size:1rem;letter-spacing:0.04em;text-transform:uppercase;margin:24px 0 12px 0;">
            Capabilities
        </h3>
        <ul style="color:{MUTED};font-size:0.9rem;line-height:2;">
            <li><strong style="color:{TEXT};">Failure Prediction</strong> — Binary classification (Healthy / Failure)</li>
            <li><strong style="color:{TEXT};">Health Score</strong> — 0–100 score from failure probability</li>
            <li><strong style="color:{TEXT};">Risk Tiers</strong> — Low / Medium / High / Critical</li>
            <li><strong style="color:{TEXT};">Maintenance Timeline</strong> — Days until action required</li>
            <li><strong style="color:{TEXT};">Sensor Alerts</strong> — Rule-based threshold recommendations</li>
            <li><strong style="color:{TEXT};">Analytics</strong> — EDA, heatmaps, scatter plots</li>
        </ul>

        <h3 style="color:{CYAN_LT};font-size:1rem;letter-spacing:0.04em;text-transform:uppercase;margin:24px 0 12px 0;">
            Tech Stack
        </h3>
        <div>
            <span class="tag">Python 3.10+</span>
            <span class="tag">XGBoost</span>
            <span class="tag">scikit-learn</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Plotly</span>
            <span class="tag">Pandas</span>
            <span class="tag">NumPy</span>
            <span class="tag">Joblib</span>
        </div>

        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:24px;">

        <h3 style="color:{CYAN_LT};font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;margin:0 0 16px 0;">
            Model Details
        </h3>

        <div style="margin-bottom:12px;">
            <div style="color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Algorithm</div>
            <div style="color:{TEXT};font-weight:600;font-size:0.9rem;">XGBoost Classifier</div>
        </div>
        <div style="margin-bottom:12px;">
            <div style="color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Features</div>
            <div style="color:{TEXT};font-weight:600;font-size:0.9rem;">11 (Type + 5 sensors + 5 flags)</div>
        </div>
        <div style="margin-bottom:12px;">
            <div style="color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Target</div>
            <div style="color:{TEXT};font-weight:600;font-size:0.9rem;">Machine failure (binary)</div>
        </div>
        <div style="margin-bottom:12px;">
            <div style="color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Imbalance Handling</div>
            <div style="color:{TEXT};font-weight:600;font-size:0.9rem;">scale_pos_weight</div>
        </div>
        <div style="margin-bottom:24px;">
            <div style="color:{MUTED};font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;">Split</div>
            <div style="color:{TEXT};font-weight:600;font-size:0.9rem;">80 / 20 stratified</div>
        </div>

        <hr style="border-color:{BORDER};margin:16px 0;">

        <h3 style="color:{CYAN_LT};font-size:0.85rem;letter-spacing:0.08em;text-transform:uppercase;margin:0 0 14px 0;">
            Author
        </h3>
        <a href="https://www.linkedin.com/in/theamitrawat/" target="_blank"
           style="display:flex;align-items:center;gap:8px;text-decoration:none;
                  color:#38BDF8;font-size:0.88rem;font-weight:600;margin-bottom:10px;">
            <img src="https://img.icons8.com/color/18/linkedin.png"/> LinkedIn
        </a>
        <a href="https://github.com/theamitrawat" target="_blank"
           style="display:flex;align-items:center;gap:8px;text-decoration:none;
                  color:{MUTED};font-size:0.88rem;font-weight:600;">
            <img src="https://img.icons8.com/ios-glyphs/18/9CA3AF/github.png"/> GitHub
        </a>

        </div>
        """, unsafe_allow_html=True)
