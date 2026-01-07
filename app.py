import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="WHR GA2 – Happiness Prediction", layout="wide")

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
    df = pd.read_csv(path)
    return df


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

    return model, (r2, rmse, mae), coef_df


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")

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

test_size = st.sidebar.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed (reproducibility)", value=42, step=1)

st.title("World Happiness Report (2020–2024) – Predictive & Descriptive Dashboard")
st.caption("GA2 Data Product: Multiple Linear Regression + Interactive Insights")

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
model, (r2, rmse, mae), coef_df = train_model(df, test_size=test_size, random_state=int(random_state))

# -----------------------------
# Layout: 2 columns
# -----------------------------
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Dataset Overview")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10), use_container_width=True)

with colB:
    st.subheader("Model Performance (Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("MAE", f"{mae:.3f}")

    st.markdown(
        """
        **Interpretation (quick):**
        - **R²**: proportion of variance explained by the predictors  
        - **RMSE/MAE**: prediction error (lower is better)
        """
    )

st.divider()

# -----------------------------
# Coefficients
# -----------------------------
st.subheader("Regression Coefficients")
st.write("Positive coefficient → increases predicted happiness (holding other variables constant).")
st.dataframe(coef_df, use_container_width=True)

st.divider()

# -----------------------------
# Prediction Tool
# -----------------------------
st.subheader("Happiness Score Prediction Tool")
st.write("Adjust the socioeconomic indicators below to get a predicted happiness score.")

# Build sliders based on data range
slider_cols = st.columns(3)
inputs = {}

for i, feat in enumerate(FEATURE_COLS):
    min_v = float(df[feat].min())
    max_v = float(df[feat].max())
    mean_v = float(df[feat].mean())

    with slider_cols[i % 3]:
        inputs[feat] = st.slider(
            feat.replace("Explained by: ", ""),
            min_value=min_v,
            max_value=max_v,
            value=mean_v,
        )

input_df = pd.DataFrame([inputs])
pred = float(model.predict(input_df)[0])

st.success(f"**Predicted Happiness Score:** {pred:.3f}")

st.caption(
    "Note: This prediction is based on a linear regression model trained on WHR 2020–2024 data. "
    "It is intended for educational and exploratory purposes."
)
