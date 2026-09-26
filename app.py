"""
Credit Card Fraud Detection — Streamlit App
--------------------------------------------
Loads a trained model (fraud_model.json) and scaler (scaler.pkl) produced by the
training notebook, and lets a user:
  1. Upload a CSV or Excel file of transactions and get fraud predictions for all of them
  2. Inspect a results dashboard (fraud count, probability distribution)
  3. If the uploaded file has a 'Class' column, see live model performance metrics
  4. Manually test a single transaction by entering Amount/Time and pasting V1-V28

Run with:
    streamlit run app.py

Expects fraud_model.json and scaler.pkl in the same folder (created by the
training notebook's "Save the Final Model" cell).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

EXPECTED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Loaded via XGBoost's native format (not pickle) - this avoids
        # "input stream corrupted" errors caused by xgboost version
        # mismatches between the training environment (e.g. Colab) and
        # wherever this app runs.
        model = xgb.XGBClassifier()
        model.load_model("fraud_model.json")
        scaler = joblib.load("scaler.pkl")  # dict: {"amount": ..., "time": ...}
        return model, scaler, None
    except (FileNotFoundError, xgb.core.XGBoostError) as e:
        return None, None, str(e)


model, scaler, load_error = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.caption("Upload transactions, get fraud predictions, and inspect model performance.")

if load_error:
    st.error(
        f"Couldn't load the model/scaler: {load_error}\n\n"
        "Run the training notebook's final cell (\"Save the Final Model\") and place "
        "the resulting `fraud_model.json` and `scaler.pkl` next to `app.py` "
        "before launching this app."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Amount/Time the same way the training pipeline did."""
    df = df.copy()
    df["Amount"] = scaler["amount"].transform(df[["Amount"]])
    df["Time"] = scaler["time"].transform(df[["Time"]])
    return df


def predict(df: pd.DataFrame, threshold: float):
    X = df[EXPECTED_COLUMNS]
    X_scaled = preprocess(X)
    # Reorder columns to exactly match the order the model was trained on,
    # since XGBoost validates column order, not just column names.
    booster_features = model.get_booster().feature_names
    if booster_features:
        X_scaled = X_scaled[booster_features]
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Probability above which a transaction is flagged as fraud. "
         "Lower it to catch more fraud at the cost of more false alarms."
)

mode = st.sidebar.radio("Mode", ["Batch upload (CSV/Excel)", "Single transaction"])


# ---------------------------------------------------------------------------
# Mode 1: Batch upload
# ---------------------------------------------------------------------------
if mode == "Batch upload (CSV/Excel)":
    st.subheader("Upload transactions")
    st.write(
        "CSV or Excel file must contain columns: `V1`...`V28`, `Amount`, `Time`. "
        "An optional `Class` column (0/1 ground truth) unlocks performance metrics."
    )
    

    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        if uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)

        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Heuristic warning: if ground truth is present and the file is large enough
        # to plausibly be the full training data (not just a held-out test split),
        # flag it so the reported metrics aren't mistaken for genuine generalization.
        if "Class" in df.columns and len(df) > 100_000:
            st.warning(
                f"This file has {len(df):,} rows with a `Class` column - large enough "
                "that it may be the full dataset the model was trained on, not a "
                "held-out test set. If so, the performance metrics below will look "
                "better than the model's real-world accuracy. Use "
                "`test_transactions.csv` from the notebook for a trustworthy result.",
                icon="⚠️"
            )

        pred, proba = predict(df, threshold)
        results = df.copy()
        results["fraud_probability"] = proba
        results["prediction"] = np.where(pred == 1, "Fraud", "Legit")

        st.success(f"Scored {len(results):,} transactions.")

        # --- Summary metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total transactions", f"{len(results):,}")
        col2.metric("Flagged as fraud", f"{int(pred.sum()):,}")
        col3.metric("Flagged rate", f"{pred.mean()*100:.2f}%")

        # --- Probability distribution ---
        st.subheader("Fraud probability distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(proba, bins=50, ax=ax)
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
        ax.set_xlabel("Predicted fraud probability")
        ax.legend()
        st.pyplot(fig)

        # --- Results table ---
        st.subheader("Results")
        st.dataframe(
            results.sort_values("fraud_probability", ascending=False).head(200),
            use_container_width=True
        )

        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full results as CSV",
            data=csv_out,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        # --- Performance metrics, if ground truth available ---
        if "Class" in df.columns:
            st.subheader("Model performance on this data")
            y_true = df["Class"]

            roc_auc = roc_auc_score(y_true, proba)
            pr_auc = average_precision_score(y_true, proba)

            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{roc_auc:.4f}")
            c2.metric("PR-AUC (Average Precision)", f"{pr_auc:.4f}")

            st.text("Classification report:")
            report_dict = classification_report(
                y_true, pred, target_names=["Legit", "Fraud"], output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.round(4)
            # support column should read as whole transaction counts, not decimals
            report_df["support"] = report_df["support"].astype(int)
            st.dataframe(report_df, use_container_width=True)

            fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
            cm = confusion_matrix(y_true, pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                        xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
            axes[0].set_title("Confusion Matrix")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")

            precision, recall, _ = precision_recall_curve(y_true, proba)
            axes[1].plot(recall, precision)
            axes[1].set_title("Precision-Recall Curve")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")

            plt.tight_layout()
            st.pyplot(fig2)


# ---------------------------------------------------------------------------
# Mode 2: Single transaction
# ---------------------------------------------------------------------------
else:
    st.subheader("Test a single transaction")
    st.write(
        "Enter the transaction's Amount and Time, and paste the 28 `V1`-`V28` "
        "values (these are always contiguous and in order in the source CSV, "
        "so you can copy them straight across without reordering anything)."
    )

    example = st.checkbox("Show an example input")
    if example:
        st.code(
            "-1.36,-0.07,2.54,1.38,-0.34,0.46,0.24,0.10,0.36,0.09,"
            "-0.55,-0.62,-0.99,-0.31,1.47,-0.47,0.21,0.03,0.40,0.25,"
            "-0.02,0.28,-0.11,0.07,0.13,-0.19,0.13,-0.02",
            language="text"
        )
        st.caption("Example Amount: 149.62  |  Example Time: 0")

    col_a, col_t = st.columns(2)
    amount_input = col_a.number_input("Amount", min_value=0.0, value=0.0, step=1.0)
    time_input = col_t.number_input(
        "Time (seconds elapsed since first transaction in dataset)",
        min_value=0.0, value=0.0, step=1.0
    )

    raw = st.text_area("Paste V1 through V28 here (28 comma-separated values)", height=100)

    if st.button("Predict"):
        # Clean up common copy-paste issues: smart/unicode minus signs, non-breaking
        # spaces, and stray line breaks from wrapped text.
        cleaned = (
            raw.replace("\u2212", "-")
               .replace("\u2013", "-")
               .replace("\u00a0", " ")
               .replace("\n", ",")
        )
        tokens = [t.strip() for t in cleaned.split(",") if t.strip() != ""]

        values, bad_tokens = [], []
        for t in tokens:
            try:
                values.append(float(t))
            except ValueError:
                bad_tokens.append(t)

        if bad_tokens:
            st.error(f"Couldn't parse these as numbers: {bad_tokens}")
        elif len(values) != 28:
            st.error(f"Expected 28 values (V1-V28), got {len(values)}.")
        else:
            full_row = values + [amount_input, time_input]
            row = pd.DataFrame([full_row], columns=EXPECTED_COLUMNS)
            pred, proba = predict(row, threshold)

            p = float(proba[0])
            label = "🚨 FRAUD" if pred[0] == 1 else "✅ Legit"

            st.metric("Prediction", label)
            st.metric("Fraud probability", f"{p*100:.2f}%")
            st.progress(min(max(p, 0.0), 1.0))


st.sidebar.markdown("---")
st.sidebar.caption(
    "Built for a credit card fraud detection. "
    "Model trained on the Kaggle 'Credit Card Fraud Detection' dataset."
)
