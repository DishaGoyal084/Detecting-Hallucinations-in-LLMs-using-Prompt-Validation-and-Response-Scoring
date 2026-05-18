# ============================================================
# HALLUCINATION DETECTION USING PROMPT ENGINEERING
# ============================================================

import pandas as pd
import streamlit as st
import time

from difflib import SequenceMatcher

# ============================================================
# IMPORT RESPONSES
# ============================================================

from responses import (
    baseline_responses,
    refined_responses
)

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Hallucination Detection System",
    layout="wide"
)

st.title("🧠 Hallucination Detection using Prompt Engineering")

st.write(
    "Compare baseline prompts vs refined prompts for hallucination reduction."
)

# ============================================================
# SIMILARITY FUNCTION
# ============================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()

# ============================================================
# HALLUCINATION DETECTION
# ============================================================

def is_hallucinated(response, truth, threshold=0.45):

    score = similarity(response, truth)

    if score >= threshold:
        return False, score

    return True, score

# ============================================================
# FILE UPLOAD
# ============================================================

file = st.file_uploader(
    "📂 Upload Dataset CSV",
    type=["csv"]
)

# ============================================================
# DATASET FORMAT
# ============================================================

st.markdown("""
### 📌 Dataset Format

| question | ground_truth | category |
|----------|--------------|----------|
| What does SQL stand for? | Structured Query Language | factual |
""")

# ============================================================
# MAIN EXECUTION
# ============================================================

if file:

    try:

        df = pd.read_csv(file)

    except Exception as e:

        st.error(f"Dataset Error: {str(e)}")

        st.stop()

    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

    required_columns = [
        "question",
        "ground_truth",
        "category"
    ]

    for col in required_columns:

        if col not in df.columns:

            st.error(f"Missing column: {col}")

            st.stop()

    # ========================================================
    # CLEAN DATA
    # ========================================================

    df = df.dropna()

    st.success("✅ Dataset Loaded Successfully")

    st.write(f"Total Questions: {len(df)}")

    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    if st.button("🚀 Run Experiment"):

        results = []

        progress_bar = st.progress(0)

        status = st.empty()

        total = len(df)

        baseline_h_count = 0

        refined_h_count = 0

        # ====================================================
        # LOOP THROUGH DATASET
        # ====================================================

        for i, row in df.iterrows():

            question = str(row["question"]).strip()

            truth = str(row["ground_truth"]).strip()

            category = str(row["category"]).strip()

            status.write(
                f"Processing {i+1}/{total}..."
            )

            # =================================================
            # GET RESPONSES
            # =================================================

            baseline_response = baseline_responses.get(
                question,
                "No Response"
            )

            refined_response = refined_responses.get(
                question,
                "No Response"
            )

            time.sleep(1)

            # =================================================
            # HALLUCINATION CHECK
            # =================================================

            baseline_h, baseline_score = is_hallucinated(
                baseline_response,
                truth
            )

            refined_h, refined_score = is_hallucinated(
                refined_response,
                truth
            )

            # =================================================
            # COUNTS
            # =================================================

            if baseline_h:

                baseline_h_count += 1

            if refined_h:

                refined_h_count += 1

            # =================================================
            # SAVE RESULTS
            # =================================================

            results.append({

                "Question": question,

                "Category": category,

                "Ground Truth": truth,

                "Baseline Response": baseline_response,

                "Refined Response": refined_response,

                "Baseline Similarity":
                    round(baseline_score, 2),

                "Refined Similarity":
                    round(refined_score, 2),

                "Baseline Hallucination":
                    baseline_h,

                "Refined Hallucination":
                    refined_h
            })

            # =================================================
            # UPDATE PROGRESS
            # =================================================

            progress_bar.progress(
                int((i + 1) / total * 100)
            )

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        status.success(
            "✅ Experiment Completed"
        )

        result_df = pd.DataFrame(results)

        st.subheader("📊 Experiment Results")

        st.dataframe(result_df)

        # ====================================================
        # METRICS
        # ====================================================

        baseline_rate = (
            baseline_h_count / total
        ) * 100

        refined_rate = (
            refined_h_count / total
        ) * 100

        improvement = (
            baseline_rate - refined_rate
        )

        # ====================================================
        # METRIC DISPLAY
        # ====================================================

        st.subheader(
            "📉 Hallucination Analysis"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🔴 Baseline Hallucination Rate",
                f"{baseline_rate:.2f}%"
            )

        with col2:

            st.metric(
                "🟢 Refined Hallucination Rate",
                f"{refined_rate:.2f}%"
            )

        with col3:

            st.metric(
                "✅ Improvement",
                f"{improvement:.2f}%"
            )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            label="⬇ Download Results CSV",

            data=csv,

            file_name="hallucination_results.csv",

            mime="text/csv"
        )

        # ====================================================
        # FINAL MESSAGE
        # ====================================================

        if refined_rate < baseline_rate:

            st.success(
                "✅ Prompt Engineering Reduced Hallucinations"
            )

        else:

            st.warning(
                "⚠ No Significant Improvement Observed"
            )