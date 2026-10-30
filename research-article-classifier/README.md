# Research Article Classification

A Streamlit demo that predicts whether a research abstract belongs to **Science**, **Business**, or **Social Science**.

## Project structure

```text
research-article-classifier/
├── app.py
├── requirements.txt
├── README.md
├── model_comparison.csv
├── model_metrics.json
├── data/
│   └── labelled_dataset.csv
├── models/
│   ├── best_model.joblib
│   ├── tfidf.joblib
│   └── classifier_bundle.joblib
├── notebooks/
│   └── Project1_research_article_classification.ipynb
└── src/
    ├── __init__.py
    └── preprocessing.py
```

## Dataset

The three supplied CSV files were combined, labeled using their source category, and cleaned using the same logic as the notebook. Rows without abstracts were removed. Exact duplicate abstracts were removed with `keep=False`, matching the notebook.

- Total unique abstracts: **1,760**
- Science: **582**
- Business: **598**
- Social Science: **580**

The combined labeled dataset is in `data/labelled_dataset.csv`.

## Models compared

The notebook's six algorithms were retrained using the same 80/20 stratified split (`random_state=42`) and TF-IDF settings. The SVM was selected because it achieved the highest test accuracy in this run.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| SVM | 0.8722 | 0.8722 | 0.8717 | 0.8719 |
| Logistic Regression | 0.8494 | 0.8515 | 0.8491 | 0.8498 |
| Naive Bayes | 0.8352 | 0.8390 | 0.8344 | 0.8348 |
| Random Forest | 0.8210 | 0.8192 | 0.8199 | 0.8186 |
| kNN | 0.7557 | 0.7518 | 0.7547 | 0.7498 |
| Decision Tree | 0.7017 | 0.7071 | 0.7016 | 0.7032 |

### Selected model test results

- Accuracy: **87.22%**
- Macro precision: **87.22%**
- Macro recall: **87.17%**
- Macro F1-score: **87.19%**

These are results on the held-out test split, not a guarantee of performance on new sources or topics.

## Run locally

1. Clone or download this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push the complete project folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose your repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

Keep the `models/`, `src/`, and `requirements.txt` paths unchanged.

## Using the demo

- **Single abstract:** paste one abstract and click **Predict category**.
- **Upload CSV:** upload a CSV with a column named exactly `Abstract`, then click **Classify uploaded dataset**.
- Download the resulting CSV using the download button.

## Preprocessing and model artifacts

The app applies the same preprocessing rules as the notebook before calling the saved TF-IDF vectorizer and LinearSVC classifier. Do not replace the preprocessing logic or vectorizer independently of the trained model.

- `models/best_model.joblib`: trained LinearSVC classifier
- `models/tfidf.joblib`: fitted TF-IDF vectorizer
- `models/classifier_bundle.joblib`: combined convenience bundle
- `src/preprocessing.py`: preprocessing function used by the app

## Data and academic integrity note

The input CSV files were supplied for this project and appear to be Scopus exports. Confirm that your use and redistribution of the abstracts complies with your institution's access terms, database license, and publisher copyright conditions before making the dataset public on GitHub. If public redistribution is not permitted, keep the dataset private or exclude it from the public repository and provide an authorized download/source description instead.

The classifier is an academic baseline. It predicts from abstract text and can make mistakes, especially for interdisciplinary research.
