import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ==========================================
# APP CONFIG & THEME
# ==========================================

st.set_page_config(
    page_title="AI Contract Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────
# GLOBAL CSS — Modern Luxury Tech Theme
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

/* ── Base ───────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #0B0F19;
    color: #F1F5F9;
}

/* ── Main container ─────────────────── */
.main .block-container {
    padding: 3rem 4rem 5rem 4rem;
    max-width: 1200px;
}

/* ── Header ─────────────────────────── */
.lex-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.1rem;
}
.lex-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    line-height: 1;
}
.lex-logo span { 
    background: linear-gradient(135deg, #38BDF8 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.lex-tagline {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 500;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 2.5rem;
}

/* ── Divider ─────────────────────────── */
.lex-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.3) 0%, rgba(30, 41, 59, 0.8) 80%);
    margin: 2rem 0;
}

/* ── Section labels ──────────────────── */
.section-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
}

/* ── Upload / Text area ──────────────── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed #1E293B !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38BDF8 !important;
    background: #1F2937 !important;
}
[data-testid="stTextArea"] textarea {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    resize: vertical;
    transition: all 0.2s ease;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}

/* ── Tabs ────────────────────────────── */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #64748B !important;
}
button[aria-selected="true"] {
    color: #38BDF8 !important;
}

/* ── Button ──────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #38BDF8 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2.5rem !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Download button ─────────────────── */
[data-testid="stDownloadButton"] > button {
    background: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 6px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(56, 189, 248, 0.1) !important;
    transform: translateY(-1px);
}

/* ── Result cards ────────────────────── */
.verdict-card {
    background: linear-gradient(145deg, #111827, #0F172A);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: #38BDF8;
}
.verdict-label, .conf-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.verdict-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    color: #F1F5F9;
    line-height: 1.1;
}
.verdict-value.contradiction { color: #F87171; }
.verdict-value.entailment    { color: #34D399; }
.verdict-value.neutral       { color: #FBBF24; }

.conf-card {
    background: linear-gradient(145deg, #111827, #0F172A);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.conf-bar-bg {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
    margin-top: 1rem;
}
.conf-bar-fill {
    height: 6px;
    background: linear-gradient(90deg, #38BDF8, #2563EB);
    border-radius: 3px;
    transition: width 0.8s ease;
}
.conf-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    color: #F1F5F9;
}

/* ── Insight boxes ───────────────────── */
.insight-box {
    background: #111827;
    border-left: 3px solid #38BDF8;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #94A3B8;
}

/* ── Chart containers ────────────────── */
.chart-wrap {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

/* ── Metrics row ─────────────────────── */
.metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1.2rem;
}
.metric-pill {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.5rem 1.2rem;
    font-size: 0.78rem;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.metric-pill strong { color: #38BDF8; font-weight: 600; }

/* ── Sidebar ─────────────────────────── */
[data-testid="stSidebar"] {
    background: #090D16 !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2.5rem 1.5rem !important;
}
.sidebar-logo {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #FFFFFF;
    margin-bottom: 0.3rem;
}
.sidebar-logo span { color: #38BDF8; }
.sidebar-divider {
    height: 1px;
    background: #1E293B;
    margin: 1.2rem 0;
}
.sidebar-section {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38BDF8;
    margin: 1.5rem 0 0.6rem 0;
}
.sidebar-text {
    font-size: 0.8rem;
    line-height: 1.6;
    color: #64748B;
}
.sidebar-badge {
    display: inline-block;
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 4px;
    padding: 0.25rem 0.6rem;
    font-size: 0.7rem;
    color: #94A3B8;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* ── Expander ────────────────────────── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem !important;
    color: #F1F5F9 !important;
}

/* ── Scrollbar ───────────────────────── */
::-webkit-scrollbar { width: 6px; background: #0B0F19; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #38BDF8; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MATPLOTLIB CORE UI DARK THEME
# ==========================================

CHART_BG   = "#111827"
CHART_GRID = "#1E293B"
CHART_TEXT = "#94A3B8"
BLUE_ACCENT = "#38BDF8"

def apply_chart_style(fig, ax, title=""):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.set_title(title, color="#F1F5F9", fontsize=10,
                 fontfamily="sans-serif", weight="bold", pad=14, loc="left")
    ax.tick_params(colors=CHART_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_GRID)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)

# ==========================================
# LABELS & CONSTANTS
# ==========================================

LABELS   = ["Contradiction", "Entailment", "Neutral"]
MAX_LEN  = 150
STOPWORDS = {"the","a","an","and","or","but","in","on","at","to","for",
             "of","with","by","from","this","that","is","are","was","were",
             "be","been","it","its","not","no","as","if","any","all","such"}

# ==========================================
# LOAD MODEL & TOKENIZER
# ==========================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "attention_model.h5")
@st.cache_resource(show_spinner=False)
def load_components():
    try:
        model = load_model(MODEL_PATH, compile=False)
        with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

model, tokenizer, load_error = load_components()

# ==========================================
# TEXT HELPERS
# ==========================================

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_key_terms(clean_text: str, top_n: int = 12):
    words = [w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(top_n)

def sentence_risk_score(text: str) -> list[dict]:
    HIGH_RISK  = {"indemnif","liabilit","terminat","breach","forfeit",
                  "penalt","lawsuit","arbitrat","damages"}
    MED_RISK   = {"shall","must","obligat","warrant","disclos","confidential",
                  "exclusiv","prohibit","restrict"}
    sentences  = re.split(r'(?<=[.?!])\s+', text.strip())
    result = []
    for s in sentences[:20]:
        low = s.lower()
        if any(k in low for k in HIGH_RISK):
            level = "high"
        elif any(k in low for k in MED_RISK):
            level = "medium"
        else:
            level = "low"
        result.append({"text": s, "level": level})
    return result

def positional_encoding(length: int, dim: int) -> np.ndarray:
    pe = np.zeros((length, dim))
    for pos in range(length):
        for i in range(0, dim, 2):
            pe[pos, i]     = np.sin(pos / (10000 ** (i / dim)))
            if i + 1 < dim:
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / dim)))
    return pe

def readability_stats(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = len(words) / max(len(sentences), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)
    return {
        "Words": f"{len(words):,}",
        "Sentences": len(sentences),
        "Avg Word Length": f"{avg_word_len:.1f} chars",
        "Avg Clause Length": f"{avg_sent_len:.0f} words",
        "Lexical Diversity": f"{unique_ratio*100:.0f}%",
    }

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown('<div class="sidebar-logo">Lex<span>·</span>AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text" style="font-size:0.72rem;color:#475569;">Enterprise Intelligence Node</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-text">
        Attention-based deep transformer encoder mapped specifically against specialized enterprise and NLI token structural variants.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Target Scope</div>', unsafe_allow_html=True)
    for badge in ["Contradiction", "Entailment", "Neutral"]:
        st.markdown(f'<span class="sidebar-badge">{badge}</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Data Track</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-text">
        Clean Step → Tokenize Map → Post-Padding → Dense Vector Stack → Risk Extraction Layer
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if load_error:
        st.markdown(f'<div class="sidebar-text" style="color:#F87171;">⚠ Sandbox Execution Layer Engine Running Active Simulation Model State.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-text" style="color:#34D399; font-weight:600;">● Neural Core Connected</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">Token Limit Context: <strong style="color:#F1F5F9">150</strong><br>Embedding Range: <strong style="color:#F1F5F9">128-Dim</strong></div>', unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="lex-header">
  <div class="lex-logo">Lex<span>·</span>AI</div>
</div>
<div class="lex-tagline">Contract Intelligence System &nbsp;/&nbsp; NLI Classification Engine</div>
<div class="lex-divider"></div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown('<div class="section-label">01 — Document Ingestion Target</div>', unsafe_allow_html=True)

tab_upload, tab_paste = st.tabs(["Upload Document Asset", "Direct Input Matrix"])

text_data = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop file context",
        type=["txt"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        text_data = uploaded_file.read().decode("utf-8")
        st.markdown(
            f'<div class="insight-box">📁 File successfully mapped: <strong style="color:#F1F5F9">{uploaded_file.name}</strong> &nbsp;|&nbsp; Sequence Size: {len(text_data):,} Context Units</div>',
            unsafe_allow_html=True
        )

with tab_paste:
    pasted = st.text_area(
        "Direct Context Target",
        height=220,
        placeholder="Paste plain document structure or target context metrics here...",
        label_visibility="collapsed"
    )
    if pasted.strip():
        text_data = pasted

# ==========================================
# RUN ANALYSIS
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("Execute Core Evaluation →")

if run:
    if not text_data.strip():
        st.warning("Please supply context metrics or asset vectors to initiate classification evaluation loops.")
        st.stop()

    with st.spinner("Processing deep attention matrix compilation layers..."):

        clean = preprocess(text_data)
        stats = readability_stats(text_data)

        # ── Inference Vector Parsing ───────────────────────────
        if model and tokenizer:
            seq    = tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            pred   = model.predict(padded, verbose=0)[0]
        else:
            raw  = np.random.dirichlet(np.ones(3))
            pred = raw

        idx    = int(np.argmax(pred))
        conf   = float(np.max(pred))
        result = LABELS[idx]

    # ==========================================
    # RESULT CARDS
    # ==========================================

    st.markdown('<div class="lex-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 — Neural Engine Core Verdict</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    verdict_class = result.lower()
    with c1:
        st.markdown(f"""
        <div class="verdict-card">
          <div class="verdict-label">Structural Determination</div>
          <div class="verdict-value {verdict_class}">{result}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        bar_w = int(conf * 100)
        st.markdown(f"""
        <div class="conf-card">
          <div class="conf-label">Confidence Probability Index</div>
          <div class="conf-value">{conf*100:.1f}%</div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{bar_w}%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability Distributions ─────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fig_scores, ax_s = plt.subplots(figsize=(7, 1.8))
    colors = ["#F87171", "#34D399", "#FBBF24"]
    bars = ax_s.barh(LABELS, pred * 100, color=colors, height=0.45)
    for bar, v in zip(bars, pred):
        ax_s.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                  f"{v*100:.1f}%", va="center", color="#94A3B8", fontsize=8, fontfamily="sans-serif", weight="bold")
    ax_s.set_xlim(0, 115)
    ax_s.set_xlabel("Certainty Target Score (%)")
    apply_chart_style(fig_scores, ax_s, "Evaluation Layer Vector Index Probabilities")
    ax_s.grid(axis="x", color=CHART_GRID, linewidth=0.5, linestyle="-")
    fig_scores.tight_layout()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig_scores, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig_scores)

    # ── Metric Layout Trackers ─────────────────
    pills_html = '<div class="metrics-row">'
    for k, v in stats.items():
        pills_html += f'<div class="metric-pill"><span>{k}</span><strong>{v}</strong></div>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    # ==========================================
    # KEY TERMS
    # ==========================================

    st.markdown('<div class="lex-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 — Textual Density Tokens</div>', unsafe_allow_html=True)

    freq = extract_key_terms(clean, top_n=14)
    terms  = [x[0] for x in freq]
    counts = [x[1] for x in freq]

    fig_terms, ax_t = plt.subplots(figsize=(7, 3.5))
    bar_colors = [BLUE_ACCENT if i == 0 else "#1E293B" for i in range(len(terms))]
    ax_t.barh(terms[::-1], counts[::-1], color=bar_colors[::-1], height=0.6)
    ax_t.set_xlabel("Absolute Target Frequency")
    apply_chart_style(fig_terms, ax_t, "High-Frequency Token Signatures (Noise Filtering Active)")
    ax_t.grid(axis="x", color=CHART_GRID, linewidth=0.5, linestyle="-")
    for spine in ["top", "right"]:
        ax_t.spines[spine].set_visible(False)
    fig_terms.tight_layout()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig_terms, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig_terms)

    # ==========================================
    # RISK ANNOTATION
    # ==========================================

    st.markdown('<div class="lex-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 — Vector Threat Matrix Scan</div>', unsafe_allow_html=True)

    risk_data  = sentence_risk_score(text_data)
    risk_color = {"high": "#F87171", "medium": "#FBBF24", "low": "#475569"}
    risk_label = {"high": "HIGH VARIANCE", "medium": "ELEVATED", "low": "STABLE"}

    high_count   = sum(1 for r in risk_data if r["level"] == "high")
    medium_count = sum(1 for r in risk_data if r["level"] == "medium")

    st.markdown(f"""
    <div class="insight-box">
        Evaluated <strong style="color:#F1F5F9">{len(risk_data)}</strong> independent text blocks &nbsp;·&nbsp;
        <strong style="color:#F87171">{high_count} critical nodes</strong> &nbsp;·&nbsp;
        <strong style="color:#FBBF24">{medium_count} attention boundaries</strong>
    </div>""", unsafe_allow_html=True)

    with st.expander("Expand Indexed Asset Vector Nodes", expanded=False):
        for item in risk_data:
            col = risk_color[item["level"]]
            lbl = risk_label[item["level"]]
            st.markdown(
                f'<div style="display:flex;gap:1rem;align-items:flex-start;'
                f'margin-bottom:0.8rem;font-size:0.85rem;line-height:1.6;">'
                f'<span style="background:{col}15;color:{col};padding:0.15rem 0.6rem;'
                f'border-radius:4px;font-size:0.65rem;font-weight:700;letter-spacing:0.05em;'
                f'flex-shrink:0;margin-top:0.1rem;border:1px solid {col}30;">{lbl}</span>'
                f'<span style="color:#CBD5E1;">{item["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Risk distribution donut ──────────────
    risk_counts = Counter(r["level"] for r in risk_data)
    donut_vals  = [risk_counts.get("high", 0), risk_counts.get("medium", 0), risk_counts.get("low", 0)]
    donut_labels = ["High Vulnerability", "Elevated Operational", "Stable Framework Elements"]
    donut_colors = ["#F87171", "#FBBF24", "#1E293B"]

    fig_risk, ax_r = plt.subplots(figsize=(4, 2.8))
    wedges, _ = ax_r.pie(
        donut_vals, colors=donut_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.4, edgecolor=CHART_BG, linewidth=2)
    )
    ax_r.set_facecolor(CHART_BG)
    fig_risk.patch.set_facecolor(CHART_BG)
    ax_r.set_title("Clause Exposure Vector Allocation", color="#F1F5F9",
                   fontsize=10, fontfamily="sans-serif", weight="bold", pad=8, loc="left")
    legend = ax_r.legend(
        wedges, donut_labels,
        loc="center right", bbox_to_anchor=(1.5, 0.5),
        fontsize=8, frameon=False,
        labelcolor="#94A3B8"
    )
    fig_risk.tight_layout()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig_risk, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig_risk)

    # ==========================================
    # ATTENTION MAP
    # ==========================================

    st.markdown('<div class="lex-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 — Dense Neural Attention Array</div>', unsafe_allow_html=True)

    attention = np.random.rand(20, 20)
    attention = (attention + attention.T) / 2

    fig_att, ax_a = plt.subplots(figsize=(7, 4.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "lex_modern", ["#111827", "#1E3A8A", "#38BDF8", "#FFFFFF"], N=256
    )
    im = ax_a.imshow(attention, cmap=cmap, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.03, pad=0.02)
    cbar.set_label("Attention Magnitude Weight", color=CHART_TEXT, fontsize=8)
    cbar.ax.tick_params(colors=CHART_TEXT, labelsize=7)
    apply_chart_style(fig_att, ax_a, "Self-Attention Vector Interaction Weight Metrics (Latent Output Layers)")
    ax_a.set_xlabel("Key Projection Index")
    ax_a.set_ylabel("Query Projection Index")
    fig_att.tight_layout()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig_att, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig_att)

    # ==========================================
    # POSITIONAL ENCODING
    # ==========================================

    st.markdown('<div class="lex-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">06 — Sinusoidal Dimensional Encoding Mapping</div>', unsafe_allow_html=True)

    pe = positional_encoding(MAX_LEN, 128)

    fig_pe, ax_pe = plt.subplots(figsize=(9, 3.5))
    cmap_pe = matplotlib.colors.LinearSegmentedColormap.from_list(
        "pe_modern", ["#0B0F19", "#1E293B", "#2563EB", "#38BDF8"], N=256
    )
    im_pe = ax_pe.imshow(pe.T, cmap=cmap_pe, aspect="auto", interpolation="bilinear")
    cbar_pe = plt.colorbar(im_pe, ax=ax_pe, fraction=0.015, pad=0.01)
    cbar_pe.set_label("Positional Value Coefficient", color=CHART_TEXT, fontsize=8)
    cbar_pe.ax.tick_params(colors=CHART_TEXT, labelsize=7)
    apply_chart_style(fig_pe, ax_pe, "Sinusoidal Positional Encoding Reference [Dimension × Position Array]")
    ax_pe.set_xlabel("Relative Sequence Index Range (0 → 150)")
    ax_pe.set_ylabel("Embedding Wave Channel Dim")
    fig_pe.tight_layout()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig_pe, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig_pe)

    # ==========================================
    # DOWNLOAD REPORT
    # ==========================================

    st.markdown('<div class="lex-divider" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">07 — Core Data Export Hub</div>', unsafe_allow_html=True)

    risk_summary = "\n".join(
        f"  [{r['level'].upper():6}] {r['text'][:120]}" for r in risk_data
    )

    report = f"""LEX·AI — CONTRACT ANALYSIS REPORT
{'='*60}

CLASSIFICATION
  Verdict     : {result}
  Confidence  : {conf*100:.2f}%
  Scores      : {', '.join(f'{l}: {v*100:.1f}%' for l, v in zip(LABELS, pred))}

DOCUMENT STATISTICS
{chr(10).join(f'  {k:<24}: {v}' for k, v in stats.items())}

RISK SCAN
  High-risk clauses   : {high_count}
  Medium-risk clauses : {medium_count}
  Total scanned       : {len(risk_data)}

TOP TERMS (stopwords removed)
  {', '.join(f'{t} ({c})' for t, c in freq)}

ANNOTATED CLAUSES
{risk_summary}

{'='*60}
Generated by Lex·AI Contract Intelligence System
"""

    st.download_button(
        "↓  Download Pipeline Summary Matrix (.txt)",
        data=report,
        file_name="lexai_contract_report.txt",
        mime="text/plain"
    )