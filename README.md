# Credit Card Fraud Detection — Web App

A Streamlit app that loads your trained model and lets you score transactions
for fraud, either in bulk (file upload) or one at a time.

## Setup

1. Make sure you have `fraud_model.json` and `scaler.pkl` — these are created by
   the "Save the Final Model" cell of the training notebook
   (`credit_card_fraud_detection.ipynb`). The model is saved in XGBoost's native
   JSON format rather than pickle, so it loads correctly even if your local
   XGBoost version differs from Colab's.
2. Put `app.py`, `requirements.txt`, `fraud_model.json`, and `scaler.pkl` all in
   the same folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. It will open in your browser at `http://localhost:8501`.

## Getting a file to test with

The notebook's Section 3.1 ("Exporting the Held-Out Test Set") exports
`test_transactions.csv` — the 20% of data the model never saw during training,
in the exact raw format the app expects. **Use this file, not the full
`creditcard.csv`**, whenever you want the app to report genuine, trustworthy
performance metrics. Uploading the full dataset will inflate the numbers,
since the model has already seen those exact transactions during training —
the app will warn you if this looks like what's happening.

## Features

- **Batch upload (CSV or Excel):** upload a file of transactions (`V1`-`V28`,
  `Amount`, `Time` columns), get fraud probability + prediction for every row,
  view a probability distribution chart, and download the results.
- **Ground-truth metrics:** if your uploaded file also has a `Class` column
  (0 = legit, 1 = fraud), the app automatically shows ROC-AUC, PR-AUC, a
  classification report (rendered as a proper table), confusion matrix, and
  precision-recall curve.
- **Data-leakage warning:** if a file with a `Class` column has over 100,000
  rows, the app warns that it may be the full training set rather than a
  held-out test set, since that would make the reported metrics misleadingly
  high.
- **Single transaction mode:** enter `Amount` and `Time` in two number fields,
  and paste the 28 `V1`-`V28` values (always contiguous and in order in the
  source CSV, so no reordering needed) to get an instant prediction — handy
  for a live demo in your viva.
- **Adjustable threshold:** a sidebar slider lets you move the fraud decision
  threshold and see how it trades off precision vs recall in real time.

## Deploying it publicly (optional, for your submission)

The easiest free option is **Streamlit Community Cloud**:

1. Push this folder (`app.py`, `requirements.txt`, `fraud_model.json`, `scaler.pkl`)
   to a GitHub repo. You don't need to include `creditcard.csv` or
   `test_transactions.csv` — those are just for testing, not required to run the app.
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app", and
   point it at your repo with `app.py` as the main file.
3. You'll get a public URL (e.g. `https://your-app-name.streamlit.app`) you can
   include in your project report.

Note: keep `fraud_model.json` under GitHub's file size limits (usually fine —
these models are typically under a few MB). The free tier "sleeps" the app
after inactivity; it wakes up automatically within a few seconds of the next
visit.

## Notes

- The threshold slider only affects the *prediction label* (Fraud/Legit) — the
  underlying probability score doesn't change, so you can freely explore
  trade-offs without re-running anything.
- Since `V1`-`V28` are PCA-transformed features (not human-readable), the
  easiest way to test the single-transaction mode is to copy those 28 values
  directly from a row in your dataset's CSV, then enter that row's `Amount`
  and `Time` separately in the two number fields above the paste box.