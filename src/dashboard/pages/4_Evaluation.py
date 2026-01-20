"""
RAGAS Evaluation Dashboard for measuring RAG system quality.
"""

import streamlit as st
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from src.dashboard.llm import LlmChain
from src.dashboard.evaluation import (
    RAGASEvaluator,
    RAGEvaluationPipeline,
    EvaluationSample,
    print_evaluation_report
)

st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 RAGAS Evaluation Dashboard")
st.markdown("""
Evaluate your RAG system using RAGAS (Retrieval Augmented Generation Assessment) metrics:
- **Faithfulness**: How factually consistent is the answer with the context?
- **Answer Relevancy**: How relevant is the answer to the question?
- **Context Precision**: Are relevant contexts ranked higher?
- **Context Recall**: Does the context contain all necessary information?
- **Context Relevancy**: How relevant are the retrieved contexts?
""")

# Initialize components
if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = LlmChain()

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

# Sidebar
with st.sidebar:
    st.markdown("### Evaluation Settings")

    eval_model = st.selectbox(
        "Evaluation Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="Model used for LLM-as-judge evaluation"
    )

    st.markdown("### Quick Evaluation")
    num_questions = st.slider("Number of test questions", 3, 10, 5)

    if st.button("🚀 Run Quick Evaluation", type="primary"):
        with st.spinner("Running evaluation... This may take a few minutes."):
            try:
                evaluator = RAGASEvaluator(model=eval_model)
                pipeline = RAGEvaluationPipeline(
                    st.session_state.llm_chain,
                    evaluator
                )

                # Default test questions
                test_questions = [
                    "What are the main topics covered in this course?",
                    "What are the prerequisites for this class?",
                    "How is the final grade calculated?",
                    "What is the late submission policy?",
                    "Who is the instructor and what are the office hours?",
                    "What textbooks or materials are required?",
                    "When are the exams scheduled?",
                    "What is the attendance policy?",
                    "How can I contact the teaching assistants?",
                    "What are the learning objectives of this course?",
                ][:num_questions]

                report = pipeline.evaluate_questions(test_questions)
                st.session_state.evaluation_results = report

                # Save report
                output_dir = Path("data/evaluations")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                evaluator.save_report(report, output_path)

                st.success(f"Evaluation complete! Report saved to {output_path}")

            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Results", "🔬 Custom Evaluation", "📜 History"])

with tab1:
    if st.session_state.evaluation_results:
        report = st.session_state.evaluation_results
        avg = report.average_scores

        # Overall score
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg.overall_score * 100,
                title={'text': "Overall Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffcccb"},
                        {'range': [40, 70], 'color': "#fffacd"},
                        {'range': [70, 100], 'color': "#90EE90"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Individual metrics
        st.markdown("### Metric Breakdown")
        col1, col2, col3, col4, col5 = st.columns(5)

        metrics = [
            ("Faithfulness", avg.faithfulness),
            ("Answer Relevancy", avg.answer_relevancy),
            ("Context Precision", avg.context_precision),
            ("Context Recall", avg.context_recall),
            ("Context Relevancy", avg.context_relevancy),
        ]

        for col, (name, value) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                delta_color = "normal" if value >= 0.7 else "inverse"
                st.metric(
                    name,
                    f"{value:.1%}",
                    delta=f"{'Good' if value >= 0.7 else 'Needs Work'}",
                    delta_color=delta_color
                )

        # Radar chart
        st.markdown("### Performance Radar")
        categories = ['Faithfulness', 'Answer\nRelevancy', 'Context\nPrecision',
                      'Context\nRecall', 'Context\nRelevancy']
        values = [avg.faithfulness, avg.answer_relevancy, avg.context_precision,
                  avg.context_recall, avg.context_relevancy]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='Current',
            line_color='#667eea',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))

        # Add target line
        fig.add_trace(go.Scatterpolar(
            r=[0.7] * 6,
            theta=categories + [categories[0]],
            fill='toself',
            name='Target (70%)',
            line_color='green',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line_dash='dash'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-question results
        st.markdown("### Per-Question Results")
        results_data = []
        for i, (sample, result) in enumerate(zip(report.samples, report.results)):
            results_data.append({
                'Question': sample.question[:50] + '...',
                'Faithfulness': result.faithfulness,
                'Answer Rel.': result.answer_relevancy,
                'Context Prec.': result.context_precision,
                'Context Rec.': result.context_recall,
                'Context Rel.': result.context_relevancy,
                'Overall': result.overall_score
            })

        df = pd.DataFrame(results_data)

        # Style the dataframe
        def color_score(val):
            if isinstance(val, float):
                if val >= 0.7:
                    return 'background-color: #90EE90'
                elif val >= 0.4:
                    return 'background-color: #fffacd'
                else:
                    return 'background-color: #ffcccb'
            return ''

        styled_df = df.style.applymap(color_score, subset=[
            'Faithfulness', 'Answer Rel.', 'Context Prec.',
            'Context Rec.', 'Context Rel.', 'Overall'
        ]).format({
            'Faithfulness': '{:.1%}',
            'Answer Rel.': '{:.1%}',
            'Context Prec.': '{:.1%}',
            'Context Rec.': '{:.1%}',
            'Context Rel.': '{:.1%}',
            'Overall': '{:.1%}'
        })

        st.dataframe(styled_df, use_container_width=True)

    else:
        st.info("👈 Run an evaluation from the sidebar to see results")

with tab2:
    st.markdown("### Custom Evaluation")
    st.markdown("Test specific questions with optional ground truth answers.")

    # Custom question input
    custom_question = st.text_area(
        "Enter your test question:",
        placeholder="What is the deadline for the final project?"
    )

    ground_truth = st.text_area(
        "Ground truth answer (optional):",
        placeholder="The final project is due on December 15th at 11:59 PM."
    )

    if st.button("Evaluate Question"):
        if custom_question:
            with st.spinner("Evaluating..."):
                try:
                    evaluator = RAGASEvaluator(model=eval_model)
                    pipeline = RAGEvaluationPipeline(
                        st.session_state.llm_chain,
                        evaluator
                    )

                    # Create sample
                    sample = pipeline.create_sample_from_query(
                        custom_question,
                        ground_truth if ground_truth else None
                    )

                    # Display RAG response
                    st.markdown("#### Generated Answer")
                    st.info(sample.answer)

                    st.markdown("#### Retrieved Context")
                    for i, ctx in enumerate(sample.contexts):
                        with st.expander(f"Context {i+1}"):
                            st.write(ctx)

                    # Evaluate
                    result = evaluator.evaluate_sample(sample)

                    st.markdown("#### Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Faithfulness", f"{result.faithfulness:.1%}")
                        st.metric("Answer Relevancy", f"{result.answer_relevancy:.1%}")
                    with col2:
                        st.metric("Context Precision", f"{result.context_precision:.1%}")
                        st.metric("Context Recall", f"{result.context_recall:.1%}")
                    with col3:
                        st.metric("Context Relevancy", f"{result.context_relevancy:.1%}")
                        st.metric("Overall Score", f"{result.overall_score:.1%}")

                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
        else:
            st.warning("Please enter a question to evaluate")

with tab3:
    st.markdown("### Evaluation History")

    # Load historical evaluations
    eval_dir = Path("data/evaluations")
    if eval_dir.exists():
        eval_files = sorted(eval_dir.glob("evaluation_*.json"), reverse=True)

        if eval_files:
            for eval_file in eval_files[:10]:  # Show last 10
                with st.expander(f"📄 {eval_file.name}"):
                    try:
                        with open(eval_file) as f:
                            data = json.load(f)

                        st.markdown(f"**Timestamp:** {data.get('timestamp', 'N/A')}")
                        st.markdown(f"**Samples:** {data.get('num_samples', 'N/A')}")

                        if 'average_scores' in data:
                            scores = data['average_scores']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall", f"{scores.get('overall_score', 0):.1%}")
                            with col2:
                                st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.1%}")
                            with col3:
                                st.metric("Relevancy", f"{scores.get('answer_relevancy', 0):.1%}")

                        # Download button
                        st.download_button(
                            "Download Report",
                            json.dumps(data, indent=2),
                            file_name=eval_file.name,
                            mime="application/json"
                        )

                    except Exception as e:
                        st.error(f"Failed to load: {e}")
        else:
            st.info("No evaluation history found")
    else:
        st.info("No evaluations have been run yet")

# Footer with recommendations
st.markdown("---")
st.markdown("### 📋 Recommendations")

if st.session_state.evaluation_results:
    avg = st.session_state.evaluation_results.average_scores
    recommendations = []

    if avg.faithfulness < 0.7:
        recommendations.append("**Faithfulness is low**: Consider improving your prompt to emphasize staying grounded in the context.")

    if avg.answer_relevancy < 0.7:
        recommendations.append("**Answer relevancy needs work**: The model may be including irrelevant information. Try more focused prompts.")

    if avg.context_precision < 0.7:
        recommendations.append("**Context precision is low**: Your retrieval is returning some irrelevant documents. Consider adjusting the similarity threshold or reranking parameters.")

    if avg.context_recall < 0.7:
        recommendations.append("**Context recall is low**: The retriever may be missing relevant information. Try increasing `initial_k` or adjusting chunk sizes.")

    if avg.context_relevancy < 0.7:
        recommendations.append("**Context relevancy needs improvement**: Consider using better embeddings or hybrid search weights.")

    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("🎉 All metrics are above the 70% threshold! Your RAG system is performing well.")
else:
    st.info("Run an evaluation to get personalized recommendations")
