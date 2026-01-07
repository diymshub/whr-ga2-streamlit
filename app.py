import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page config + light styling
# --------------------------------------------------
st.set_page_config(page_title="WHR GA2 – Happiness Prediction", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {padding-top: 1rem;}
h1, h2, h3 {letter-spacing: 0.2px;}
[data-testid="stMetric"] {background: rgba(255,255,255,0.04); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Constants
# --------------------------------------------------
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

# --------------------------------------------------
# Data + model helpers
# --------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame, test_size=0.2, random_state=42):
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

    coef_df = pd.DataFrame({
        "Variable": FEATURE_COLS,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    return model, (r2, rmse, mae), coef_df, results_df


def find_closest_by_score(df: pd.DataFrame, pred_score: float, top_k: int = 5):
    tmp = df[["Country name", "Year", TARGET_COL]].copy()
    tmp["Difference"] = (tmp[TARGET_COL] - pred_score).abs()
    return tmp.sort_values("Difference").head(top_k)


def find_closest_by_profile(df: pd.DataFrame, input_df: pd.DataFrame, top_k: int = 5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    input_scaled = scaler.transform(input_df[FEATURE_COLS])

    distances = np.sqrt(((X_scaled - input_scaled) ** 2).sum(axis=1))

    tmp = df[["Country name", "Year", TARGET_COL]].copy()
    tmp["Profile distance"] = distances
    return tmp.sort_values("Profile distance").head(top_k)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔧 Controls")
st.sidebar.caption("Data source & modelling settings")
st.sidebar.divider()

data_source = st.sidebar.radio(
    "Data source",
    ["Use repository CSV", "Upload CSV"],
    index=0
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload WHR cleaned CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)
else:
    df = load_data(DEFAULT_CSV)

st.sidebar.divider()
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("World Happiness Report (2020–2024)")
st.caption("GA2 Deployed Data Product — Predictive Happiness Modelling")

# --------------------------------------------------
# Validation
# --------------------------------------------------
missing_cols = [c for c in [TARGET_COL] + FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# --------------------------------------------------
# Train model
# --------------------------------------------------
model, (r2, rmse, mae), coef_df, results_df = train_model(
    df, test_size=test_size, random_state=int(random_state)
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🧠 Model", "🧮 Predict", "ℹ️ About"]
)

# -------------------- Overview --------------------
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Years", f"{int(df['Year'].min())}–{int(df['Year'].max())}")

    st.dataframe(df.head(12), use_container_width=True)

# -------------------- Model --------------------
with tab2:
    st.subheader("Model Performance (Hold-out Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("MAE", f"{mae:.3f}")

    st.divider()
    st.subheader("Regression Coefficients")
    st.dataframe(coef_df, use_container_width=True)

    st.divider()
    st.subheader("Actual vs Predicted")
    st.line_chart(results_df.reset_index(drop=True), use_container_width=True)

# -------------------- Predict --------------------
with tab3:
    st.subheader("Happiness Score Prediction")
    st.write("Adjust indicators and click **Predict**.")

    with st.form("predict_form"):
        cols = st.columns(3)
        inputs = {}

        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 3]:
                inputs[feat] = st.slider(
                    feat.replace("Explained by: ", ""),
                    float(df[feat].min()),
                    float(df[feat].max()),
                    float(df[feat].mean())
                )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])
        pred = float(model.predict(input_df)[0])

        st.success(f"**Predicted Happiness Score:** {pred:.3f}")

        st.divider()
        st.subheader("Equivalent Country–Year Examples")

        by_score = find_closest_by_score(df, pred)
        by_profile = find_closest_by_profile(df, input_df)

        st.markdown("**A) Closest by happiness score**")
        st.dataframe(by_score, use_container_width=True)

        st.markdown("**B) Closest by socioeconomic profile**")
        st.dataframe(by_profile, use_container_width=True)

        top = by_score.iloc[0]
        st.info(
            f"This score is most similar to **{top['Country name']} ({int(top['Year'])})**, "
            f"with an actual happiness score of **{top[TARGET_COL]:.3f}**."
        )

# -------------------- About --------------------
with tab4:
    st.subheader("About This Application")
    st.markdown("""
**Course:** WQD7001 — Principles of Data Science (GA2)  
**Dataset:** World Happiness Report (2020–2024)

**Method**
- Predictors: GDP, social support, health, freedom, generosity, corruption
- Model: Multiple Linear Regression
- Evaluation: R², RMSE, MAE (test set)

**Purpose**
This app allows exploratory scenario analysis by linking socioeconomic indicators
to predicted happiness scores.

**SDGs**
- SDG 3: Good Health & Well-being  
- SDG 10: Reduced Inequalities
""")
