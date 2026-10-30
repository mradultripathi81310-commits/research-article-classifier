import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text

MODEL_PATH = ROOT / "models" / "best_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf.joblib"

st.set_page_config(
    page_title="Research Article Classifier",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Make sure the models/ folder is included in the repository."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_abstracts(abstracts, model, vectorizer):
    cleaned = [clean_text(value) for value in abstracts]
    features = vectorizer.transform(cleaned)
    return model.predict(features)

st.title("Research Article Classification")
st.caption("Classify research abstracts into Science, Business, or Social Science.")
st.markdown(
    "This demo uses NLP preprocessing, TF-IDF features, and a trained Linear Support Vector Machine (SVM)."
)

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()

classes = list(model.classes_)
c1, c2, c3 = st.columns(3)
c1.metric("Available categories", len(classes))
c2.metric("Training dataset", "1,760 abstracts")
c3.metric("Test accuracy", "87.22%")

tab_single, tab_batch, tab_about = st.tabs(
    ["Single abstract", "Upload CSV", "About the model"]
)

with tab_single:
    st.subheader("Predict one research abstract")
    abstract = st.text_area(
        "Paste the research abstract",
        height=220,
        placeholder="Paste the abstract text here...",
        key="single_abstract",
    )
    if st.button("Predict category", type="primary", key="single_predict"):
        if not abstract.strip():
            st.warning("Please enter an abstract before predicting.")
        else:
            prediction = predict_abstracts([abstract], model, vectorizer)[0]
            st.success(f"Predicted category: **{prediction}**")
            st.caption("This is a model prediction, not a peer-review or publication-quality assessment.")

with tab_batch:
    st.subheader("Classify multiple abstracts")
    st.write("Upload a CSV file containing a column named `Abstract`.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded, encoding="utf-8-sig")
            if "Abstract" not in input_df.columns:
                st.error("The uploaded CSV must contain a column named exactly `Abstract`.")
            else:
                valid_mask = input_df["Abstract"].notna() & input_df["Abstract"].astype(str).str.strip().ne("")
                if not valid_mask.any():
                    st.warning("No non-empty abstracts were found in the uploaded file.")
                else:
                    if st.button("Classify uploaded dataset", type="primary"):
                        output_df = input_df.copy()
                        output_df["Predicted Label"] = "Not classified: empty abstract"
                        predictions = predict_abstracts(
                            output_df.loc[valid_mask, "Abstract"].astype(str).tolist(),
                            model,
                            vectorizer,
                        )
                        output_df.loc[valid_mask, "Predicted Label"] = predictions
                        st.success(f"Classified {int(valid_mask.sum())} abstract(s).")
                        st.dataframe(output_df, use_container_width=True)
                        st.download_button(
                            "Download predictions CSV",
                            data=output_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="research_article_predictions.csv",
                            mime="text/csv",
                        )
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

with tab_about:
    st.subheader("Model information")
    st.markdown(
        """
        - **Task:** three-class text classification
        - **Classes:** Science, Business, Social Science
        - **Text features:** TF-IDF, up to 5,000 features, unigrams and bigrams
        - **Classifier:** LinearSVC with balanced class weights
        - **Train/test split:** 80% / 20%, stratified, random state 42
        - **Evaluation:** accuracy, macro precision, macro recall, macro F1-score
        """
    )
    st.info(
        "The SVM decision function is not a calibrated probability. "
        "This demo therefore displays the predicted class without presenting an uncalibrated score as confidence."
    )
    st.write("Model classes:", ", ".join(classes))

st.divider()
st.caption("Academic baseline project • Research Article Classification")
