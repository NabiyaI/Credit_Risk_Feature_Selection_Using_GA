"""Streamlit frontend for GA-based SVM feature selection on German credit data."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

from ga_svm_feature_selection import run_experiment

APP_DIR = Path(__file__).resolve().parent

GERMAN_DATA_PATH = Path("german_credit_data.csv")
SECONDARY_DATA_PATH = Path("credit_risk_dataset.csv")


def _resolve_input_path(path_text: str) -> Path:
    """Resolve a user-provided path.

    - Absolute paths are used as-is.
    - Relative paths are resolved relative to this app's directory.
    """
    p = Path(path_text)
    return p if p.is_absolute() else (APP_DIR / p)


def _run_pipeline_with_logs(
    csv_path: Path,
    allow_proxy_target: bool,
    dataset_mode: Literal["german", "generic"],
) -> tuple[dict, str]:
    """Run experiment and capture console logs (generation progress)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = run_experiment(
            csv_path=csv_path,
            allow_proxy_target=allow_proxy_target,
            dataset_mode=dataset_mode,
        )
    return result, buffer.getvalue()


def _render_result_block(title: str, result: dict, logs: str) -> None:
    """Render one dataset result section."""
    st.markdown(f"## {title}")

    target_source = result["target_source"]
    if target_source == "proxy":
        st.warning(
            "No target column was found. The run used a proxy risk label. "
            "Use a true label column for final research conclusions."
        )
    elif target_source == "uci_download":
        st.info(
            "No target column was found in your CSV, so labels were auto-loaded "
            "from the official UCI German Credit dataset (row-order matched)."
        )
    else:
        st.success(f"Target source detected: {target_source}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.4f}")
    col2.metric("Traditional Accuracy", f"{result['traditional_accuracy']:.4f}")
    col3.metric("Final GA Accuracy", f"{result['final_accuracy']:.4f}")
    col4.metric("GA Features Reduced", str(result["reduced_count"]))

    st.markdown("### Side-by-Side Method Comparison")
    comparison_df = pd.DataFrame(result["comparison_table"])
    st.dataframe(comparison_df, use_container_width=True)

    efficiency_df = comparison_df.copy()
    efficiency_df["Feature Reduction %"] = (
        (result["total_count"] - efficiency_df["Selected Features"]) / result["total_count"] * 100
    ).round(2)
    efficiency_df["Accuracy per Feature"] = (
        efficiency_df["10-fold CV Accuracy"] / efficiency_df["Selected Features"]
    ).round(5)

    st.markdown("### Accuracy Comparison")
    chart_view = efficiency_df.copy()
    chart_view["Method Display"] = chart_view["Method"].replace(
        {
            "Baseline (All Features)": "Baseline - All Features",
            "Traditional (SelectKBest)": "Traditional - SelectKBest",
            "GA Wrapper (SVM Fitness)": "GA Wrapper - SVM Fitness",
        }
    )

    acc_chart = (
        alt.Chart(chart_view)
        .mark_bar(color="#1f77b4")
        .encode(
            y=alt.Y(
                "Method Display:N",
                sort="-x",
                title="Method",
                axis=alt.Axis(labelLimit=1000, labelFontSize=13),
            ),
            x=alt.X("10-fold CV Accuracy:Q", title="10-fold CV Accuracy", scale=alt.Scale(domain=[0.0, 1.0])),
            tooltip=[
                alt.Tooltip("Method:N"),
                alt.Tooltip("10-fold CV Accuracy:Q", format=".4f"),
                alt.Tooltip("Selected Features:Q"),
                alt.Tooltip("Feature Reduction %:Q", format=".2f"),
                alt.Tooltip("Accuracy per Feature:Q", format=".5f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(acc_chart, use_container_width=True)

    st.markdown("### Efficiency View")
    st.dataframe(
        efficiency_df[
            [
                "Method",
                "Selected Features",
                "Feature Reduction %",
                "10-fold CV Accuracy",
                "Accuracy per Feature",
                "Accuracy Gain vs Baseline",
            ]
        ],
        use_container_width=True,
    )

    ga_row = efficiency_df.loc[efficiency_df["Method"] == "GA Wrapper (SVM Fitness)"].iloc[0]
    tr_row = efficiency_df.loc[efficiency_df["Method"] == "Traditional (SelectKBest)"].iloc[0]
    ga_better_accuracy = float(ga_row["10-fold CV Accuracy"]) > float(tr_row["10-fold CV Accuracy"])
    ga_better_efficiency = float(ga_row["Accuracy per Feature"]) > float(tr_row["Accuracy per Feature"])

    st.markdown("### GA Efficiency Verdict")
    st.write(
        {
            "GA_accuracy_vs_traditional": "higher" if ga_better_accuracy else "not higher",
            "GA_accuracy_per_feature_vs_traditional": "higher" if ga_better_efficiency else "not higher",
            "GA_accuracy_gain_over_baseline": round(float(ga_row["Accuracy Gain vs Baseline"]), 4),
            "GA_feature_reduction_percent": round(float(ga_row["Feature Reduction %"]), 2),
        }
    )

    st.markdown("### Feature Subset Summary")
    st.write(
        {
            "total_features": result["total_count"],
            "selected_features": result["selected_count"],
            "reduced_features": result["reduced_count"],
            "traditional_selected_features": result["traditional_selected_count"],
            "traditional_best_k": result["traditional_best_k"],
        }
    )

    st.markdown("### Traditional Method Features (SelectKBest)")
    traditional_df = pd.DataFrame({"feature": result["traditional_selected_features"]})
    st.dataframe(traditional_df, use_container_width=True)

    st.markdown("### Selected Features")
    selected_df = pd.DataFrame({"feature": result["selected_features"]})
    st.dataframe(selected_df, use_container_width=True)

    with st.expander("GA Generation Logs", expanded=False):
        st.text(logs if logs.strip() else "No logs captured.")


def main() -> None:
    st.set_page_config(
        page_title="GA Feature Selection for Credit Risk",
        page_icon="📊",
        layout="wide",
    )

    st.title("Genetic Algorithm-based Feature Selection")
    st.subheader("Wrapper Method: SVM + 10-Fold CV + DEAP")

    st.markdown(
        "Use this app to run GA wrapper feature selection and inspect the optimal "
        "feature subset for credit risk analysis."
    )

    with st.sidebar:
        st.header("Configuration")
        german_path_input = st.text_input("German dataset CSV", value=str(GERMAN_DATA_PATH))
        secondary_path_input = st.text_input("Second dataset CSV", value=str(SECONDARY_DATA_PATH))
        allow_proxy = st.checkbox(
            "Allow proxy target if label missing",
            value=True,
            help=(
                "If checked, a proxy risk label is generated when no explicit target "
                "column is found (for demo/testing only)."
            ),
        )
        run_btn = st.button("Run Both Datasets", type="primary")

    if not run_btn:
        st.info("Set options in the sidebar and click 'Run Both Datasets'.")
        return

    german_path = _resolve_input_path(german_path_input)
    secondary_path = _resolve_input_path(secondary_path_input)

    st.markdown("---")
    st.markdown("## Dataset 1: German Credit")
    if not german_path.exists():
        st.error(f"File not found: {german_path}")
    else:
        with st.spinner("Running Dataset 1 (German Credit)..."):
            try:
                result, logs = _run_pipeline_with_logs(
                    csv_path=german_path,
                    allow_proxy_target=allow_proxy,
                    dataset_mode="german",
                )
                _render_result_block("Results - German Credit", result, logs)
            except Exception as exc:
                st.exception(exc)

    st.markdown("---")
    st.markdown("## Dataset 2: Secondary Credit Risk Dataset")
    if not secondary_path.exists():
        st.warning(
            "Second dataset file not found. Add the CSV and rerun to compare both datasets."
        )
    else:
        with st.spinner("Running Dataset 2 (Secondary Credit Risk)..."):
            try:
                result, logs = _run_pipeline_with_logs(
                    csv_path=secondary_path,
                    allow_proxy_target=allow_proxy,
                    dataset_mode="generic",
                )
                _render_result_block("Results - Secondary Credit Risk", result, logs)
            except Exception as exc:
                st.exception(exc)


if __name__ == "__main__":
    main()
