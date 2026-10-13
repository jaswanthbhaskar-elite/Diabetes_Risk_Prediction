"""
Diabetes Risk Prediction — Streamlit App
-----------------------------------------
Loads the already-trained Logistic Regression and Random Forest
scikit-learn pipelines (produced in the Colab notebook) and uses them
to predict diabetes risk from user-entered patient data.

This app does NOT retrain anything. It only loads:
    - logistic_regression_pipeline.pkl
    - random_forest_pipeline.pkl
    - feature_metadata.pkl
and passes user input through the saved pipelines exactly as they
were used during training/evaluation.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

MODEL_DIR = Path(__file__).parent

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression_pipeline.pkl",
    "Random Forest": MODEL_DIR / "random_forest_pipeline.pkl",
}
METADATA_FILE = MODEL_DIR / "feature_metadata.pkl"


# ---------------------------------------------------------------------------
# Styling — a single injected CSS block. Everything below is presentation
# only; no prediction logic lives here.
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stApp {
            background-color: #F4F7F8;
        }

        /* Laptop-style dashboard width: full "wide" layout stretches too far
           on large monitors, so cap the content column at a comfortable
           laptop-screen width and center it. */
        .main .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            margin: 0 auto;
        }

        /* Force readable text everywhere. The app's own colored elements
           (buttons, result-card verdicts, etc.) use more specific selectors
           further down, so they still win over this blanket rule regardless
           of the visitor's light/dark system theme. */
        .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
            color: #16292B;
        }

        /* Widget question text, e.g. "Gender", "Age (years)", "Model" radio label */
        [data-testid="stWidgetLabel"] p {
            color: #16292B !important;
            font-weight: 500;
        }

        /* Radio button option text (Logistic Regression / Random Forest / Compare Both) */
        [data-testid="stRadio"] label p {
            color: #16292B !important;
            font-weight: 400;
        }

        /* Header */
        .app-header {
            padding: 1.75rem 0 0.5rem 0;
        }
        .app-header h1 {
            color: #16292B;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .app-header p {
            color: #5C6E71;
            font-size: 1rem;
            margin-top: 0;
        }

        /* Section labels */
        .section-label {
            color: #16292B;
            font-size: 1.05rem;
            font-weight: 600;
            margin: 1.1rem 0 0.6rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #DCE4E5;
        }

        /* Form card — styles Streamlit's own form container directly,
           instead of a manually-inserted wrapper div (which rendered as
           an empty box, since markdown output can't wrap later widgets). */
        [data-testid="stForm"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E9EA;
            border-radius: 10px;
            padding: 1.4rem 1.5rem 1rem 1.5rem;
            margin-bottom: 1rem;
        }

        /* Predict button */
        div.stButton > button, div.stFormSubmitButton > button {
            background-color: #0E6E67;
            color: #FFFFFF;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1rem;
            width: 100%;
            transition: background-color 0.15s ease;
        }
        /* The blanket ".stApp p/div/span" text-color rule above also matches
           the <p>/<div> Streamlit renders *inside* the button for its label,
           which was overriding the white button text. Force it back to white
           on every descendant of the button, not just the button element itself. */
        div.stButton > button *, div.stFormSubmitButton > button * {
            color: #FFFFFF !important;
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            background-color: #0B5751;
            color: #FFFFFF;
        }

        /* Result cards */
        .result-card {
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
            background-color: #FFFFFF;
            border: 1px solid #E2E9EA;
            border-left: 6px solid #9AA7A9;
        }
        .result-card.positive {
            border-left-color: #B5432E;
            background-color: #FBF2F0;
        }
        .result-card.negative {
            border-left-color: #2E7D5B;
            background-color: #F0F7F3;
        }
        .result-card .model-name {
            color: #5C6E71;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.2rem;
        }
        .result-card .verdict {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .result-card.positive .verdict { color: #9A3A28; }
        .result-card.negative .verdict { color: #256B4E; }
        .result-card .probability {
            color: #16292B;
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }
        .result-card .probability b {
            font-size: 1.05rem;
        }

        /* Footer */
        .app-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #DCE4E5;
            color: #7C8B8D;
            font-size: 0.82rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached loaders — models/metadata are loaded once and reused across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    """Load a single saved scikit-learn pipeline, raising a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    try:
        return joblib.load(path)
    except Exception as exc:  # corrupted file, version mismatch, etc.
        raise RuntimeError(f"Could not load '{path.name}': {exc}") from exc


@st.cache_resource(show_spinner=False)
def load_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path.name}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load '{path.name}': {exc}") from exc


@st.cache_resource(show_spinner=False)
def get_categorical_options(_pipeline):
    """
    Read the exact categories the OneHotEncoder learned during training,
    straight from the fitted pipeline, instead of hard-coding/guessing them.
    Returns a dict: {column_name: [category1, category2, ...]}
    """
    preprocessor = _pipeline.named_steps["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    cat_columns = preprocessor.transformers_[1][2]  # column names for the 'cat' block
    return {
        col: list(cats)
        for col, cats in zip(cat_columns, cat_transformer.categories_)
    }


# ---------------------------------------------------------------------------
# Load everything up front, with friendly error handling
# ---------------------------------------------------------------------------
inject_css()

load_error = None
pipelines = {}
metadata = None

try:
    metadata = load_metadata(METADATA_FILE)
    for name, path in MODEL_FILES.items():
        pipelines[name] = load_pipeline(path)
except FileNotFoundError as exc:
    load_error = f"A required file is missing. {exc}"
except RuntimeError as exc:
    load_error = str(exc)
except Exception as exc:  # anything unforeseen
    load_error = f"Unexpected error while loading models: {exc}"

if load_error:
    st.error(
        "⚠️ The application could not start because a model or metadata "
        f"file could not be loaded.\n\n**Details:** {load_error}\n\n"
        "Make sure `logistic_regression_pipeline.pkl`, "
        "`random_forest_pipeline.pkl`, and `feature_metadata.pkl` are in "
        "the same folder as `app.py`."
    )
    st.stop()

# Categorical options read directly from the fitted encoder (not guessed)
categorical_options = get_categorical_options(pipelines["Random Forest"])
GENDER_OPTIONS = categorical_options.get("gender", ["Female", "Male", "Other"])
SMOKING_OPTIONS = categorical_options.get(
    "smoking_history",
    ["No Info", "current", "ever", "former", "never", "not current"],
)

ALL_INPUT_COLUMNS = metadata.get(
    "all_input_columns",
    ["gender", "age", "hypertension", "heart_disease",
     "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"],
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🩺 Diabetes Risk Prediction</h1>
        <p>Estimates diabetes risk from patient health data using a Logistic Regression
        or Random Forest model trained on the Kaggle Diabetes Prediction Dataset
        (96,146 records after cleaning).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "Choose which trained model to use for prediction:",
    options=["Logistic Regression", "Random Forest", "Compare Both"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Patient input form
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

with st.form("patient_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=GENDER_OPTIONS)
        age = st.number_input(
            "Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0
        )
        hypertension_label = st.selectbox("Hypertension", options=["No", "Yes"])
        heart_disease_label = st.selectbox("Heart Disease", options=["No", "Yes"])

    with col2:
        smoking_history = st.selectbox("Smoking History", options=SMOKING_OPTIONS)
        bmi = st.number_input(
            "BMI", min_value=5.0, max_value=100.0, value=25.0, step=0.1
        )
        hba1c_level = st.number_input(
            "HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1
        )
        blood_glucose_level = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=50, max_value=400, value=120, step=1,
        )

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_input_dataframe():
    """Assemble one-row DataFrame with columns/types matching what the
    pipelines were trained on. Column order doesn't matter to a
    ColumnTransformer (it selects by name), but we still align it with
    feature_metadata's recorded column order for clarity."""
    row = {
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension_label == "Yes" else 0,
        "heart_disease": 1 if heart_disease_label == "Yes" else 0,
        "smoking_history": smoking_history,
        "bmi": float(bmi),
        "HbA1c_level": float(hba1c_level),
        "blood_glucose_level": int(blood_glucose_level),
    }
    return pd.DataFrame([row], columns=ALL_INPUT_COLUMNS)


def predict_with(pipeline, input_df):
    """Run a saved pipeline on the input and return (predicted_class, probability).

    predict_proba() returns one row of [P(class 0), P(class 1)] because both
    pipelines' classes_ are confirmed to be [0, 1] in that order — so index 1
    is always P(diabetes = 1), the positive-class probability we display.
    """
    pred_class = pipeline.predict(input_df)[0]
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(input_df)[0][1]  # P(class = 1)
    return pred_class, probability


def format_probability(probability):
    """Format a 0-1 probability as a percentage without silently rounding a
    small-but-real value down to '0.0%'. Only an actually-zero probability
    should ever display as 0.0%."""
    if probability is None:
        return "N/A"
    pct = probability * 100
    if pct <= 0:
        return "0.0%"
    if round(pct, 1) == 0.0:
        # Standard 1-decimal rounding would hide this real, nonzero value —
        # show more decimal places instead of reporting it as exactly zero.
        if round(pct, 4) == 0.0:
            return "<0.0001%"
        return f"{pct:.4f}%"
    return f"{pct:.1f}%"


def render_result_card(model_name, pred_class, probability):
    """Render one styled result card. pred_class == 1 means diabetes positive."""
    is_positive = pred_class == 1
    verdict_text = "Diabetes Positive" if is_positive else "Diabetes Negative"
    icon = "⚠️" if is_positive else "✅"
    css_class = "positive" if is_positive else "negative"

    proba_html = ""
    if probability is not None:
        proba_html = f'<div class="probability">Estimated probability: <b>{format_probability(probability)}</b></div>'
    else:
        proba_html = '<div class="probability">Probability not available for this model.</div>'

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="model-name">{model_name}</div>
            <div class="verdict">{icon} {verdict_text}</div>
            {proba_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
    try:
        input_df = build_input_dataframe()

        if model_choice == "Compare Both":
            col_lr, col_rf = st.columns(2)
            for col, name in zip((col_lr, col_rf), pipelines.keys()):
                pred_class, probability = predict_with(pipelines[name], input_df)
                with col:
                    render_result_card(name, pred_class, probability)
        else:
            pred_class, probability = predict_with(pipelines[model_choice], input_df)
            render_result_card(model_choice, pred_class, probability)

    except Exception as exc:
        st.error(
            "Something went wrong while making the prediction. "
            f"Details: {exc}"
        )

# ---------------------------------------------------------------------------
# Footer / disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        For educational and predictive analysis purposes only.
        This application is not a substitute for professional medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)