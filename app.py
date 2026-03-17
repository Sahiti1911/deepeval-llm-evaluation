import streamlit as st
import json
import pandas as pd
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

st.set_page_config(page_title="Evaluation UI", page_icon="📊")

st.title("📊 DeepEval Progress UI")
st.write("Run your LLM evaluations and monitor their progress in real-time.")

@st.cache_data
def load_dataset():
    with open("data.json", "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_dataset()
st.write(f"**Loaded {len(dataset)} items to evaluate.**")

if st.button("Start Evaluation"):
    # Initialize the metric
    metric = GEval(
        name="Response Quality",
        criteria="""
        Evaluate correctness, relevance, completeness, and clarity.
        Higher score = better answer.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ]
    )

    results = []
    
    # Setup UI Elements for tracking progress
    progress_bar = st.progress(0, text="Initializing evaluation...")
    status_text = st.empty()
    table_placeholder = st.empty()
    
    total = len(dataset)
    
    for i, item in enumerate(dataset):
        # Update progress text
        status_text.info(f"Evaluating item ID: {item.get('id', i)} ({i+1}/{total})...")
        
        question = item["input"]
        chatgpt_resp = item["chatgpt_response"]
        myapp_resp = item["myapp_response"]

        test_chatgpt = LLMTestCase(input=question, actual_output=chatgpt_resp)
        test_myapp = LLMTestCase(input=question, actual_output=myapp_resp)

        # Measure the scores
        try:
            score_chatgpt = metric.measure(test_chatgpt)
            score_myapp = metric.measure(test_myapp)

            if score_chatgpt > score_myapp:
                winner = "chatgpt"
            elif score_chatgpt < score_myapp:
                winner = "myapp"
            else:
                winner = "tie"
        except Exception as e:
            st.error(f"Error evaluating item {item.get('id', i)}: {e}")
            score_chatgpt = 0
            score_myapp = 0
            winner = "error"

        # Save result
        results.append({
            "id": item.get("id", i),
            "chatgpt_score": score_chatgpt,
            "myapp_score": score_myapp,
            "winner": winner
        })
        
        # Update progress bar
        progress = (i + 1) / total
        progress_bar.progress(progress, text=f"Progress: {int(progress * 100)}%")
        
        # Update the table in real-time
        df = pd.DataFrame(results)
        table_placeholder.dataframe(df, use_container_width=True)

    status_text.success("✅ Evaluation complete!")
    
    # Final Summary section
    st.subheader("Results Summary")
    
    if results:
        df = pd.DataFrame(results)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("ChatGPT Wins", len(df[df['winner'] == 'chatgpt']))
        col2.metric("MyApp Wins", len(df[df['winner'] == 'myapp']))
        col3.metric("Ties", len(df[df['winner'] == 'tie']))
        
        st.bar_chart(df['winner'].value_counts())
