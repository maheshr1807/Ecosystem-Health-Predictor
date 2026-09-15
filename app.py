import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ecosystem Health Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — vibrant "nature tech" theme
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a2f23 0%, #2d5a3d 50%, #1e3d2f 100%);
    }
    h1, h2, h3, h4 {
        color: #f0fff4 !important;
        font-family: 'Trebuchet MS', sans-serif;
    }
    p, label, span, div {
        color: #e6f5ea;
    }
    .hero-banner {
        background: linear-gradient(90deg, #f7b733 0%, #fc4a1a 100%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero-banner h1 {
        color: #331a05 !important;
        margin: 0;
        font-size: 2.3rem;
    }
    .hero-banner p {
        color: #4a2408 !important;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(6px);
    }
    .result-healthy {
        background: linear-gradient(120deg, #56ab2f, #a8e063);
        color: #1b3306;
        padding: 26px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(86,171,47,0.4);
        animation: pulseGreen 1.8s ease-in-out infinite;
    }
    .result-atrisk {
        background: linear-gradient(120deg, #f7b733, #fc4a1a);
        color: #3d1c02;
        padding: 26px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(247,183,51,0.4);
    }
    .result-degraded {
        background: linear-gradient(120deg, #8e0e00, #1f1c18);
        color: #ffe4e0;
        padding: 26px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(142,14,0,0.4);
    }
    @keyframes pulseGreen {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2317 0%, #1c3a26 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #56ab2f, #a8e063);
        color: #1b3306;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 26px;
        font-size: 1.05rem;
        transition: 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(168,224,99,0.45);
    }
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.06);
        border-radius: 10px 10px 0 0;
        color: #e6f5ea !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🌿 Ecosystem Health Predictor</h1>
        <p>Powered by a Gaussian Naive Bayes model &nbsp;•&nbsp;
        Classify an ecosystem as Healthy, At Risk, or Degraded from environmental readings</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# LOAD PRE-TRAINED MODEL + REFERENCE DATA (no training here)
# ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "ecosystem.pkl"
DATA_PATH = "ecosystem_data.csv"

LABEL_MAP = {0: "Healthy", 1: "At Risk", 2: "Degraded"}
LABEL_COLORS = {"Healthy": "#a8e063", "At Risk": "#f7b733", "Degraded": "#8e0e00"}
FEATURES = ["water_quality", "air_quality_index", "biodiversity_index", "vegetation_cover", "soil_ph"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_reference_data(path):
    df = pd.read_csv(path)
    df["label"] = df["ecosystem_health"].map({"healthy": "Healthy", "at risk": "At Risk", "degraded": "Degraded"})
    return df


model = load_model(MODEL_PATH)
ref_data = load_reference_data(DATA_PATH)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — inputs
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧮 Input Environmental Readings")

    water_quality = st.slider(
        "💧 Water Quality",
        float(ref_data.water_quality.min()), float(ref_data.water_quality.max()),
        float(ref_data.water_quality.mean()), step=0.1,
    )
    air_quality_index = st.slider(
        "🌫️ Air Quality Index",
        float(ref_data.air_quality_index.min()), float(ref_data.air_quality_index.max()),
        float(ref_data.air_quality_index.mean()), step=0.5,
    )
    biodiversity_index = st.slider(
        "🦋 Biodiversity Index",
        float(ref_data.biodiversity_index.min()), float(ref_data.biodiversity_index.max()),
        float(ref_data.biodiversity_index.mean()), step=0.01,
    )
    vegetation_cover = st.slider(
        "🌳 Vegetation Cover (%)",
        float(ref_data.vegetation_cover.min()), float(ref_data.vegetation_cover.max()),
        float(ref_data.vegetation_cover.mean()), step=0.1,
    )
    soil_ph = st.slider(
        "🧪 Soil pH",
        float(ref_data.soil_ph.min()), float(ref_data.soil_ph.max()),
        float(ref_data.soil_ph.mean()), step=0.1,
    )

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Ecosystem Health")

# ────────────────────────────────────────────────────────────────────────────
# TOP METRIC ROW
# ────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="metric-card"><h3>📊 Records</h3>
        <h2 style="color:#a8e063 !important;">{len(ref_data):,}</h2></div>""",
        unsafe_allow_html=True,
    )
with c2:
    healthy_pct = (ref_data.label == "Healthy").mean() * 100
    st.markdown(
        f"""<div class="metric-card"><h3>🟢 Healthy</h3>
        <h2 style="color:#a8e063 !important;">{healthy_pct:.1f}%</h2></div>""",
        unsafe_allow_html=True,
    )
with c3:
    risk_pct = (ref_data.label == "At Risk").mean() * 100
    st.markdown(
        f"""<div class="metric-card"><h3>🟠 At Risk</h3>
        <h2 style="color:#f7b733 !important;">{risk_pct:.1f}%</h2></div>""",
        unsafe_allow_html=True,
    )
with c4:
    degraded_pct = (ref_data.label == "Degraded").mean() * 100
    st.markdown(
        f"""<div class="metric-card"><h3>🔴 Degraded</h3>
        <h2 style="color:#fc4a1a !important;">{degraded_pct:.1f}%</h2></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediction", "🔍 Explore Data"])

# ---- TAB 1: PREDICTION ----
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Your Reading")
        input_df = pd.DataFrame(
            {
                "water_quality": [water_quality],
                "air_quality_index": [air_quality_index],
                "biodiversity_index": [biodiversity_index],
                "vegetation_cover": [vegetation_cover],
                "soil_ph": [soil_ph],
            }
        )
        st.dataframe(input_df.style.format(precision=2), use_container_width=True)

        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=[
                    water_quality / ref_data.water_quality.max() * 100,
                    100 - (air_quality_index / ref_data.air_quality_index.max() * 100),
                    biodiversity_index / ref_data.biodiversity_index.max() * 100,
                    vegetation_cover / ref_data.vegetation_cover.max() * 100,
                    (soil_ph - ref_data.soil_ph.min()) / (ref_data.soil_ph.max() - ref_data.soil_ph.min()) * 100,
                ],
                theta=["Water Quality", "Air Quality (inv.)", "Biodiversity", "Vegetation", "Soil pH"],
                fill="toself",
                line_color="#a8e063",
            )
        )
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="white"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=20, b=20),
            height=380,
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with right:
        st.markdown("### Prediction Result")
        if predict_clicked:
            pred_class = model.predict(input_df[FEATURES])[0]
            proba = model.predict_proba(input_df[FEATURES])[0]
            pred_label = LABEL_MAP[pred_class]

            if pred_label == "Healthy":
                st.markdown(
                    f"""<div class="result-healthy">🌿 HEALTHY ECOSYSTEM<br>
                    <span style="font-size:1rem;">Confidence: {proba[0]*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )
                st.balloons()
            elif pred_label == "At Risk":
                st.markdown(
                    f"""<div class="result-atrisk">⚠️ ECOSYSTEM AT RISK<br>
                    <span style="font-size:1rem;">Confidence: {proba[1]*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="result-degraded">🔥 DEGRADED ECOSYSTEM<br>
                    <span style="font-size:1rem;">Confidence: {proba[2]*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            labels_order = ["Healthy", "At Risk", "Degraded"]
            prob_fig = px.bar(
                x=labels_order,
                y=proba,
                color=labels_order,
                color_discrete_map=LABEL_COLORS,
                labels={"x": "Class", "y": "Probability"},
                text=[f"{p*100:.1f}%" for p in proba],
            )
            prob_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
                height=320,
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        else:
            st.info("👈 Adjust the sliders in the sidebar and click **Predict Ecosystem Health** to see results here.")

# ---- TAB 2: EXPLORE DATA ----
with tab2:
    st.markdown("### Dataset Snapshot")
    st.dataframe(ref_data.drop(columns=["label"]).head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Distributions")
        feature_choice = st.selectbox("Choose a feature", FEATURES)
        hist_fig = px.histogram(
            ref_data, x=feature_choice, color="label",
            color_discrete_map=LABEL_COLORS,
            barmode="overlay", nbins=40,
        )
        hist_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=380,
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.markdown("### Feature Relationships")
        x_axis = st.selectbox("X-axis", FEATURES, index=0, key="xaxis")
        y_axis = st.selectbox("Y-axis", FEATURES, index=2, key="yaxis")
        scatter_fig = px.scatter(
            ref_data, x=x_axis, y=y_axis,
            color="label",
            color_discrete_map=LABEL_COLORS,
            opacity=0.6,
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=380,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("### Class Balance")
    pie_fig = px.pie(
        ref_data, names="label",
        color="label",
        color_discrete_map=LABEL_COLORS,
        hole=0.45,
    )
    pie_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380,
    )
    st.plotly_chart(pie_fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Built with Streamlit • Gaussian Naive Bayes • "
    "Pre-trained model deployment 🌍</p>",
    unsafe_allow_html=True,
)