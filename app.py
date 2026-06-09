import streamlit as st
import pickle
import pandas as pd
import re

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* Force dark blue background everywhere */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp,
[data-testid="stHeader"],
section.main > div {
    background-color: #060E1F !important;
}

[data-testid="stSidebar"] {
    background-color: #0A1628 !important;
}

/* Hide default Streamlit top bar color */
[data-testid="stHeader"] {
    background: #060E1F !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0D2137 0%, #0A1E3D 60%, #071529 100%);
    border: 1px solid #1E5F8A;
    border-radius: 18px;
    padding: 40px 30px 30px 30px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 4px 32px #1E90FF22;
}

.hero-icon {
    font-size: 56px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #E8F4FD;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    line-height: 1.2;
}

.hero-title span {
    color: #1E90FF;
}

.hero-subtitle {
    font-size: 16px;
    color: #7EB8D4;
    margin-top: 6px;
    letter-spacing: 0.4px;
}

/* ── Badge row ── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.badge {
    background-color: #0D2137;
    border: 1px solid #1E5F8A;
    color: #7EB8D4;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
}

/* ── Input card ── */
.input-card {
    background-color: #0D1F35;
    border: 1px solid #1E4A6A;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
}

.section-label {
    color: #7EB8D4;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #071529 !important;
    color: #C9E8F5 !important;
    border: 1.5px solid #1E5F8A !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #1E90FF !important;
    box-shadow: 0 0 12px #1E90FF33 !important;
}
.stTextArea textarea::placeholder {
    color: #3A6080 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0 0%, #1E90FF 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3.2em !important;
    width: 100% !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    box-shadow: 0 4px 20px #1E90FF44 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1E90FF 0%, #42AAFF 100%) !important;
    box-shadow: 0 6px 28px #1E90FF66 !important;
    transform: translateY(-2px) !important;
}

/* ── Result cards ── */
.result-card {
    background: #0D1F35;
    border: 1.5px solid #1E4A6A;
    border-radius: 14px;
    padding: 20px;
    margin-top: 8px;
    text-align: center;
}

.result-fake {
    border-color: #E53935 !important;
    background: #1A0D0D !important;
}

.result-real {
    border-color: #43A047 !important;
    background: #0D1A0D !important;
}

.result-label {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}

.conf-bar-bg {
    background: #071529;
    border-radius: 20px;
    height: 10px;
    margin-top: 10px;
    overflow: hidden;
}

.conf-bar {
    height: 10px;
    border-radius: 20px;
    background: linear-gradient(90deg, #1565C0, #1E90FF);
}

/* ── Agreement box ── */
.agree-box {
    border-radius: 14px;
    padding: 18px 24px;
    margin-top: 18px;
    font-size: 17px;
    font-weight: 600;
    text-align: center;
    border: 1.5px solid;
}

/* ── Divider ── */
hr { border-color: #1E3A5F !important; opacity: 0.4; }

/* ── General text ── */
p, li, label { color: #A8D8F0 !important; }
h1, h2, h3, h4 { color: #A8D8F0 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #2E6080;
    font-size: 13px;
    margin-top: 16px;
    padding: 10px;
}

/* info/warning/error boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    background-color: #0D1F35 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🔍</div>
  <div class="hero-title">Fake News <span>Detection</span> & Optimistic System</div>
  <div class="hero-subtitle">using NLP & ML analysis to identify Fake vs Real News</div>
  <div class="badge-row">
    <span class="badge">🤖 Logistic Regression</span>
    <span class="badge">📊 Naive Bayes</span>
    <span class="badge">⚡ Real-time Analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        lr_model   = pickle.load(open("model.pkl",      "rb"))
        nb_model   = pickle.load(open("nb_model.pkl",   "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return lr_model, nb_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None, None

LR, NB, vectorization = load_models()

# ── Text preprocessing ───────────────────────────────────────
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def output_label(n):
    return "Fake News" if n == 0 else "Real News"

# ── Input Section ─────────────────────────────────────────────
st.markdown('<div class="section-label">📰 Paste Your News Article</div>', unsafe_allow_html=True)
news = st.text_area(
    label="",
    height=160,
    placeholder="Paste your news article or headline here to check if it's real or fake..."
)

col_btn, _ = st.columns([2, 1])
with col_btn:
    analyze = st.button("🔎  Analyze News")

# ── Analysis ──────────────────────────────────────────────────
if analyze:
    if news.strip() == "":
        st.warning("⚠️ Please enter some news text first.")
    elif LR is None:
        st.error("❌ Models could not be loaded. Check file paths.")
    else:
        with st.spinner("🔄 Analyzing..."):
            new_df = pd.DataFrame({"text": [news]})
            new_df["text"] = new_df["text"].apply(wordopt)
            new_xv  = vectorization.transform(new_df["text"])

            pred_lr  = LR.predict(new_xv)[0]
            pred_nb  = NB.predict(new_xv)[0]
            prob_lr  = LR.predict_proba(new_xv)[0]
            prob_nb  = NB.predict_proba(new_xv)[0]
            conf_lr  = prob_lr[pred_lr] * 100
            conf_nb  = prob_nb[pred_nb] * 100

        st.markdown("---")
        st.markdown('<div class="section-label">🎯 Prediction Results</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        for col, pred, prob, conf, name in [
            (col1, pred_lr, prob_lr, conf_lr, "Logistic Regression"),
            (col2, pred_nb, prob_nb, conf_nb, "Naive Bayes"),
        ]:
            label     = output_label(pred)
            is_fake   = pred == 0
            card_cls  = "result-fake" if is_fake else "result-real"
            emoji     = "❌" if is_fake else "✅"
            clr       = "#FF5252" if is_fake else "#69F0AE"
            bar_clr   = "#E53935" if is_fake else "#43A047"

            with col:
                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <div style="color:#7EB8D4; font-size:13px; font-weight:600;
                              letter-spacing:1px; text-transform:uppercase;
                              margin-bottom:10px;">{name}</div>
                  <div class="result-label" style="color:{clr};">{emoji} {label}</div>
                  <div style="color:#7EB8D4; font-size:14px;">
                      Confidence: <b style="color:{clr};">{conf:.1f}%</b>
                  </div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar"
                         style="width:{conf:.1f}%; background:linear-gradient(90deg,
                         {'#C62828,#EF5350' if is_fake else '#2E7D32,#66BB6A'});"></div>
                  </div>
                  <div style="color:#3A6080; font-size:12px; margin-top:10px;">
                    Fake {prob[0]*100:.1f}%  ·  Real {prob[1]*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Agreement ─────────────────────────────────────────
        if pred_lr == pred_nb:
            st.markdown(f"""
            <div class="agree-box"
                 style="background:#071D35; border-color:#1E90FF; color:#A8D8F0;">
              ✅ &nbsp;<b>Strong Prediction</b> — Both models agree:
              &nbsp;<span style="color:#1E90FF;">{output_label(pred_lr)}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agree-box"
                 style="background:#1A1A0D; border-color:#FFA726; color:#FFD54F;">
              ⚠️ &nbsp;<b>Weak Prediction</b> — Models disagree.
              LR: <b>{output_label(pred_lr)}</b> &nbsp;|&nbsp;
              NB: <b>{output_label(pred_nb)}</b>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">💡 Built with ❤️ using Machine Learning & NLP &nbsp;·&nbsp; '
    'Logistic Regression · Naive Bayes · TF-IDF</div>',
    unsafe_allow_html=True
)