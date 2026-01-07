import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="WHR GA2 – Happiness Prediction", layout="wide")

# --- Light styling (safe & subtle) ---
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {padding-top: 1rem;}
h1, h2, h3 {letter-spacing: 0.2px;}
[data-testid="stMetric"] {background: rgba(255,255,255,0.04); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants: columns
# -----------------------------
TARGET_COL = "Life evaluation (3-year average)"
FEATURE_COLS = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
]
DEFAULT_CSV = "WHRDATAFIGURE25_cleaned.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    coef_df = pd.DataFrame({"Variable": FEATURE_COLS, "Coefficient": model.coef_})
    coef_df = coef_df.sort_values("Coefficient", ascending=False).reset_index(drop=True)

    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})

    return model, (r2, rmse, mae), coef_df, results_df


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🔧 Controls")
st.sidebar.caption("Adjust data source and modelling settings.")
st.sidebar.divider()

data_source = st.sidebar.radio(
    "Data source",
    ["Use repository CSV", "Upload CSV"],
    index=0
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your WHR cleaned CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = load_data(DEFAULT_CSV)

st.sidebar.divider()
test_size = st.sidebar.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed (reproducibility)", value=42, step=1)

st.sidebar.divider()
st.sidebar.caption("Tip: Keep the seed fixed for reproducible results.")

# -----------------------------
# Header
# -----------------------------
st.title("World Happiness Report (2020–2024)")
st.caption("GA2 Data Product — Predictive Modelling (Multiple Linear Regression) + Interactive Dashboard")

# -----------------------------
# Basic validation
# -----------------------------
missing_cols = [c for c in [TARGET_COL] + FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# -----------------------------
# Train model + metrics
# -----------------------------
model, (r2, rmse, mae), coef_df, results_df = train_model(
    df, test_size=test_size, random_state=int(random_state)
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 Model", "🧮 Predict", "ℹ️ About"])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Columns", f"{df.shape[1]}")
    year_min = int(df["Year"].min()) if "Year" in df.columns else None
    year_max = int(df["Year"].max()) if "Year" in df.columns else None
    c3.metric("Year Range", f"{year_min}–{year_max}" if year_min is not None else "N/A")

    st.dataframe(df.head(12), use_container_width=True)

    with st.expander("Show column list"):
        st.write(list(df.columns))

# --- Tab 2: Model ---
with tab2:
    st.subheader("Model Performance (Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("MAE", f"{mae:.3f}")

    st.caption(
        "R² indicates variance explained; RMSE/MAE measure prediction error (lower is better). "
        "Evaluation is performed on the hold-out test set."
    )

    st.divider()

    st.subheader("Regression Coefficients")
    st.write("Coefficients represent the marginal effect **holding other predictors constant**.")
    st.dataframe(coef_df, use_container_width=True)

    st.divider()

    st.subheader("Actual vs Predicted (Hold-out Test Set)")
    st.caption("A simple visual check: predicted values should generally track the actual values.")
    st.line_chart(results_df.reset_index(drop=True), use_container_width=True)

# --- Tab 3: Predict ---
with tab3:
    st.subheader("Happiness Score Prediction Tool")
    st.write("Adjust the indicators below, then click **Predict**.")

    with st.form("predict_form"):
        cols = st.columns(3)
        inputs = {}

        for i, feat in enumerate(FEATURE_COLS):
            min_v = float(df[feat].min())
            max_v = float(df[feat].max())
            mean_v = float(df[feat].mean())

            with cols[i % 3]:
                inputs[feat] = st.slider(
                    feat.replace("Explained by: ", ""),
                    min_value=min_v,
                    max_value=max_v,
                    value=mean_v,
                )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])
        pred = float(model.predict(input_df)[0])
        st.success(f"**Predicted Happiness Score:** {pred:.3f}")

        st.caption(
            "Interpretation: this is a model-based estimate. Use it for exploration and scenario analysis."
        )
    else:
        st.info("Adjust the sliders and click **Predict** to generate a prediction.")

# --- Tab 4: About ---
with tab4:
    st.subheader("About This Data Product")
    st.markdown("""
**Course:** WQD7001 — Principles of Data Science (GA2)  
**Dataset:** World Happiness Report (2020–2024)  
**Goal:** Predict national happiness scores (life evaluation) using socioeconomic indicators.

### Method Summary
- Selected predictors: GDP per capita, social support, healthy life expectancy, freedom, generosity, and perceptions of corruption
- Train/test split with a fixed random seed for reproducibility
- Baseline model: Multiple Linear Regression
- Evaluation metrics: R², RMSE, MAE (reported on hold-out test set)

### SDG Link
- **SDG 3:** Good Health & Well-being  
- **SDG 10:** Reduced Inequalities  
""")

    st.caption(
        "Note: This application is for educational and exploratory purposes (scenario analysis), not policy automation."
    )

