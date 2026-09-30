import streamlit as st
import time

from cara_backend import cara_route
from model_manager import generate_response


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CARA | Adaptive AI Routing",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS ONLY
# NO CUSTOM HTML COMPONENTS
# ============================================================

st.markdown(
    """
<style>

/* ==============================
   MAIN BACKGROUND
   ============================== */

.stApp {
    background: #050914;
    color: #eef5ff;
}

/* Reduce top whitespace */

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* ==============================
   HIDE STREAMLIT UI
   ============================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==============================
   SIDEBAR
   ============================== */

section[data-testid="stSidebar"] {
    background: #050a16;
    border-right: 1px solid #142542;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f6ff;
}

.sidebar-title {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
}

.sidebar-subtitle {
    color: #7183a3;
    font-size: 13px;
}


/* ==============================
   HEADINGS
   ============================== */

h1 {
    color: #f5f8ff !important;
    font-weight: 800 !important;
}

h2 {
    color: #e8f1ff !important;
    font-weight: 750 !important;
}

h3 {
    color: #dce9ff !important;
}


/* ==============================
   TEXT
   ============================== */

p {
    color: #8d9db8;
}


/* ==============================
   TEXT AREA
   ============================== */

textarea {
    background-color: #071122 !important;
    color: #f4f8ff !important;
    border: 1px solid #24456f !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

textarea:focus {
    border-color: #218cff !important;
    box-shadow: 0 0 0 1px #218cff !important;
}


/* ==============================
   BUTTON
   ============================== */

.stButton > button {
    background: #087ff5 !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.2rem !important;
}

.stButton > button:hover {
    background: #1491ff !important;
}


/* ==============================
   METRIC CARDS
   ============================== */

div[data-testid="stMetric"] {
    background: #071224;
    border: 1px solid #162d4d;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #7185a5 !important;
}

div[data-testid="stMetricValue"] {
    color: #f4f8ff !important;
}


/* ==============================
   EXPANDERS
   ============================== */

div[data-testid="stExpander"] {
    background: #071224;
    border: 1px solid #172f50;
    border-radius: 10px;
}

div[data-testid="stExpander"] summary {
    color: #dceaff !important;
}


/* ==============================
   ALERTS
   ============================== */

div[data-testid="stAlert"] {
    border-radius: 10px;
}


/* ==============================
   DIVIDER
   ============================== */

hr {
    border-color: #14243b !important;
}


/* ==============================
   CAPTION
   ============================== */

.stCaption {
    color: #667993 !important;
}


/* ==============================
   SIDEBAR ITEMS
   ============================== */

section[data-testid="stSidebar"] .stMarkdown {
    color: #8fa0ba;
}


/* ==============================
   STATUS DOT
   ============================== */

.status-online {
    color: #36df79;
    font-weight: 700;
}


/* ==============================
   ROUTE BOX
   ============================== */

.route-small {
    background: #062b1d;
    border: 1px solid #126b47;
    border-radius: 12px;
    padding: 18px;
}

.route-medium {
    background: #071d38;
    border: 1px solid #18599b;
    border-radius: 12px;
    padding: 18px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🧠 CARA"
    )

    st.caption(
        "Adaptive Multi-LLM Routing"
    )

    st.divider()

    st.subheader("System Components")

    st.markdown("🟢 **Semantic Embeddings**")
    st.markdown("🔵 **Historical Evidence**")
    st.markdown("🟣 **Local Reliability**")
    st.markdown("🟡 **Confidence Estimation**")
    st.markdown("🔴 **Adaptive Routing**")

    st.divider()

    st.subheader("Available Models")

    st.info(
        "**SMALL MODEL**\n\n"
        "Qwen2.5-0.5B-Instruct"
    )

    st.info(
        "**MEDIUM MODEL**\n\n"
        "Qwen2.5-1.5B-Instruct"
    )

    st.divider()

    st.caption(
        "CARA v1.0\n"
        "Confidence-Aware Adaptive Routing"
    )


# ============================================================
# COMPACT HERO
# ============================================================

hero_left, hero_right = st.columns([4, 1])

with hero_left:

    st.markdown("# 🧠 CARA")

    st.markdown(
        "### Local Calibration-Aware Adaptive Routing"
    )

    st.caption(
        "Confidence-aware model selection for multi-LLM systems."
    )

with hero_right:

    st.markdown("### 🟢")
    st.caption("SYSTEM ONLINE")


st.divider()


# ============================================================
# QUERY SECTION
# ============================================================

st.markdown("## 💬 Query Intelligence")

st.caption(
    "Enter a query and CARA will analyze its semantic similarity, "
    "historical evidence, reliability and confidence before routing it."
)


query = st.text_area(
    "Enter your query",
    placeholder=(
        "Example: Explain how backpropagation works "
        "in a neural network."
    ),
    height=120
)


analyze = st.button(
    "🚀 Analyze & Route Query",
    use_container_width=False
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze:

    if not query.strip():

        st.warning(
            "Please enter a query first."
        )

    else:

        # ----------------------------------------------------
        # CARA ANALYSIS
        # ----------------------------------------------------

        with st.spinner(
            "CARA is analyzing the query..."
        ):

            decision = cara_route(query)


        st.success(
            "CARA analysis completed successfully."
        )


        # ----------------------------------------------------
        # INTELLIGENCE ANALYSIS
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "## 📊 Intelligence Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "LOCAL RELIABILITY",
                f"{decision['local_reliability']:.3f}"
            )

        with col2:

            st.metric(
                "EVIDENCE STRENGTH",
                f"{decision['evidence_strength']:.3f}"
            )

        with col3:

            st.metric(
                "CARA CONFIDENCE",
                f"{decision['cara_confidence']:.3f}"
            )


        # ----------------------------------------------------
        # ROUTING DECISION
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "## 🎯 Adaptive Routing Decision"
        )

        route = decision["route"]


        if route == "SMALL":

            st.success(
                "🟢 SMALL MODEL SELECTED\n\n"
                "Qwen2.5-0.5B-Instruct"
            )

        else:

            st.info(
                "🔵 MEDIUM MODEL SELECTED\n\n"
                "Qwen2.5-1.5B-Instruct"
            )


        # ----------------------------------------------------
        # DECISION STATUS
        # ----------------------------------------------------

        st.markdown(
            f"### {decision['confidence_status']}"
        )

        st.write(
            decision["decision_reason"]
        )


        # ----------------------------------------------------
        # HISTORICAL EVIDENCE
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "## 🔍 Local Historical Evidence"
        )

        st.caption(
            "Historical queries used by CARA to support the routing decision."
        )


        for i, item in enumerate(
            decision["evidence"],
            1
        ):

            outcome = (
                "Small model sufficient"
                if item["small_sufficient"] == 1
                else "Small model insufficient"
            )

            with st.expander(
                f"Evidence {i}  •  Similarity {item['similarity']:.3f}"
            ):

                st.write(
                    "**Historical Query**"
                )

                st.write(
                    item["query"]
                )

                st.write(
                    f"**Outcome:** {outcome}"
                )


        # ----------------------------------------------------
        # MODEL RESPONSE
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "## 🤖 Generated Response"
        )

        with st.spinner(
            f"Generating response using the {route} model..."
        ):

            start_time = time.time()

            answer = generate_response(
                query,
                route
            )

            response_time = (
                time.time() - start_time
            )


        st.write(answer)

        st.caption(
            f"Response generated in "
            f"{response_time:.2f} seconds "
            f"using the {route} model."
        )