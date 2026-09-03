import streamlit as st
import time

from cara_backend import cara_route
from model_manager import generate_response


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="CARA",
    page_icon="🧠",
    layout="wide"
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🧠 CARA")
st.subheader("Local Calibration-Aware Adaptive Routing for Multi-LLM Systems")

st.markdown(
    """
    CARA analyzes your query, examines similar historical queries,
    estimates local reliability, and adaptively selects the most
    suitable language model.
    """
)

st.divider()


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

query = st.text_area(
    "Enter your query",
    placeholder="Example: Explain how backpropagation works in a neural network.",
    height=120
)


# -------------------------------------------------
# ANALYZE BUTTON
# -------------------------------------------------

if st.button("🚀 Analyze & Route", use_container_width=True):

    if not query.strip():

        st.warning("Please enter a query first.")

    else:

        # -----------------------------------------
        # CARA ROUTING
        # -----------------------------------------

        with st.spinner("CARA is analyzing the query..."):

            decision = cara_route(query)

        st.success("Analysis complete!")

        st.divider()


        # -----------------------------------------
        # METRICS
        # -----------------------------------------

        st.subheader("📊 CARA Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Local Reliability",
            f"{decision['local_reliability']:.3f}"
        )

        col2.metric(
            "Evidence Strength",
            f"{decision['evidence_strength']:.3f}"
        )

        col3.metric(
            "CARA Confidence",
            f"{decision['cara_confidence']:.3f}"
        )


        # -----------------------------------------
        # ROUTING DECISION
        # -----------------------------------------

        st.divider()

        st.subheader("🎯 Routing Decision")

        route = decision["route"]

        if route == "SMALL":

            st.success(
                "🟢 SMALL MODEL SELECTED — Qwen2.5-0.5B-Instruct"
            )

        else:

            st.info(
                "🔵 MEDIUM MODEL SELECTED — Qwen2.5-1.5B-Instruct"
            )


        # -----------------------------------------
        # HISTORICAL EVIDENCE
        # -----------------------------------------

        st.divider()

        st.subheader("🔍 Local Historical Evidence")

        for i, item in enumerate(decision["evidence"], 1):

            with st.expander(
                f"Similar Query {i} — Similarity: {item['similarity']:.3f}"
            ):

                st.write("**Historical Query:**")
                st.write(item["query"])

                if item["small_sufficient"] == 1:

                    st.success(
                        "Historical Outcome: Small model was sufficient"
                    )

                else:

                    st.warning(
                        "Historical Outcome: Small model was insufficient"
                    )


        # -----------------------------------------
        # MODEL RESPONSE
        # -----------------------------------------

        st.divider()

        st.subheader("🤖 Generated Response")

        with st.spinner(
            f"Generating response using the {route} model..."
        ):

            start_time = time.time()

            answer = generate_response(
                query,
                route
            )

            response_time = time.time() - start_time


        st.write(answer)

        st.caption(
            f"Response generated in {response_time:.2f} seconds"
        )