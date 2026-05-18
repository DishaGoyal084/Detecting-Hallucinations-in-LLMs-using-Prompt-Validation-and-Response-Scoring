# =====================================================
# HALLUCINATION DETECTION RESEARCH PROJECT
# =====================================================

import pandas as pd
import streamlit as st
from google import genai
import time

from difflib import SequenceMatcher

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Hallucination Detection System",
    layout="wide"
)

st.title("🧠 Hallucination Detection using Prompt Engineering")

st.write(
    "Compare baseline prompts vs refined prompts for hallucination reduction."
)

# =====================================================
# GEMINI API CONFIGURATION
# =====================================================

# genai.configure(
#     api_key=""
# )

# model = genai.GenerativeModel(
#     "gemini-1.5-flash-latest"
# )
client = genai.Client(
    api_key="AIzaSyDyC4TU0jAM0swEvNFeITZoqP1RQiUfPX4"
)
# =====================================================
# BASELINE PROMPT
# =====================================================

def baseline_prompt(question):

    return question

# =====================================================
# REFINED PROMPT
# =====================================================

def refined_prompt(question):

    return f"""
Answer the following question accurately.

Rules:
- Give only factual information
- Do not make assumptions
- If unsure, say 'I don't know'
- Keep answer concise and clear

Question:
{question}
"""

# =====================================================
# MODEL CALL FUNCTION
# =====================================================

def call_model(prompt):

    retries = 3

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:

            error_text = str(e)

            # Rate limit handling
            if "429" in error_text:

                time.sleep(15)

                continue

            return f"API Error: {error_text}"

    return "API Error: Rate Limit Exceeded"
# =====================================================
# HALLUCINATION DETECTION
# =====================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        a.lower(),
        b.lower()
    ).ratio()

def is_hallucinated(response, truth, threshold=0.45):

    score = similarity(response, truth)

    if score >= threshold:

        return False

    return True

# =====================================================
# FILE UPLOAD
# =====================================================

file = st.file_uploader(
    "📂 Upload Dataset CSV",
    type=["csv"]
)

# =====================================================
# DATASET FORMAT
# =====================================================

st.markdown("""
### 📌 Dataset Format

Your CSV should contain:

| question | ground_truth | category |
|----------|--------------|-----------|
| Who discovered gravity? | Isaac Newton | factual |
| Why does ice float? | Ice is less dense than water | reasoning |
""")

# =====================================================
# MAIN EXECUTION
# =====================================================

if file:

    try:

        df = pd.read_csv(file)

    except Exception as e:

        st.error(f"Dataset Error: {str(e)}")

        st.stop()

    # =====================================================
    # VALIDATE REQUIRED COLUMNS
    # =====================================================

    required_columns = [
        "question",
        "ground_truth",
        "category"
    ]

    for col in required_columns:

        if col not in df.columns:

            st.error(f"Missing column: {col}")

            st.stop()

    # =====================================================
    # REMOVE EMPTY ROWS
    # =====================================================

    df = df.dropna(
        subset=[
            "question",
            "ground_truth"
        ]
    )

    st.success("✅ Dataset Loaded Successfully")

    st.write(f"Total Questions: {len(df)}")

    # =====================================================
    # RUN EXPERIMENT BUTTON
    # =====================================================

    if st.button("🚀 Run Experiment"):

        results = []

        progress_bar = st.progress(0)

        status = st.empty()

        total = len(df)

        baseline_h_count = 0

        refined_h_count = 0

        # =====================================================
        # LOOP THROUGH DATASET
        # =====================================================

        for i, row in df.iterrows():

            question = str(row["question"])

            truth = str(row["ground_truth"])

            category = str(row["category"])

            status.write(
                f"Processing {i + 1}/{total}..."
            )

            # =====================================================
            # BASELINE RESPONSE
            # =====================================================

            baseline_response = call_model(
                baseline_prompt(question)
            )

            time.sleep(8)

            # =====================================================
            # REFINED RESPONSE
            # =====================================================

            refined_response = call_model(
                refined_prompt(question)
            )

            time.sleep(2)

            # =====================================================
            # HALLUCINATION CHECK
            # =====================================================

            baseline_h = is_hallucinated(
                baseline_response,
                truth
            )

            refined_h = is_hallucinated(
                refined_response,
                truth
            )

            if baseline_h:

                baseline_h_count += 1

            if refined_h:

                refined_h_count += 1

            # =====================================================
            # SAVE RESULTS
            # =====================================================

            results.append({

                "Category": category,

                "Question": question,

                "Ground Truth": truth,

                "Baseline Response": baseline_response,

                "Refined Response": refined_response,

                "Baseline Hallucination": baseline_h,

                "Refined Hallucination": refined_h

            })

            # =====================================================
            # UPDATE PROGRESS BAR
            # =====================================================

            progress_bar.progress(
                int((i + 1) / total * 100)
            )

        # =====================================================
        # FINAL RESULTS
        # =====================================================

        status.success(
            "✅ Experiment Completed"
        )

        result_df = pd.DataFrame(results)

        st.subheader("📊 Experiment Results")

        st.dataframe(result_df)

        # =====================================================
        # METRICS
        # =====================================================

        baseline_rate = (
            baseline_h_count / total
        ) * 100

        refined_rate = (
            refined_h_count / total
        ) * 100

        improvement = (
            baseline_rate - refined_rate
        )

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

        # =====================================================
        # DOWNLOAD RESULTS
        # =====================================================

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="hallucination_results.csv",
            mime="text/csv"
        )

        # =====================================================
        # FINAL MESSAGE
        # =====================================================

        if refined_rate < baseline_rate:

            st.success(
                "✅ Prompt Engineering Reduced Hallucinations"
            )

        else:

            st.warning(
                "⚠ No Significant Improvement Observed"
            )