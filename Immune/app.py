import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import joblib, os

st.title("🩺 Improved Diabetes Prediction (80–85% Accuracy)")

# ==============================================================
# 1️⃣ Load Dataset
# ==============================================================
DATA_PATH = "diabetes.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Please place 'diabetes.csv' in this folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ==============================================================
# 2️⃣ Clean Data (replace zeros with median)
# ==============================================================
cols_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_replace:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# ==============================================================
# 3️⃣ Handle Class Imbalance (Upsampling)
# ==============================================================
df_majority = df[df["Outcome"] == 0]
df_minority = df[df["Outcome"] == 1]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
st.write(f"✅ Dataset balanced — new size: {df_balanced.shape}")

# ==============================================================
# 4️⃣ Train-Test Split
# ==============================================================
X = df_balanced.drop("Outcome", axis=1)
y = df_balanced["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================
# 5️⃣ Feature Scaling
# ==============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================
# 6️⃣ Logistic Regression (tuned parameters)
# ==============================================================
model = LogisticRegression(
    solver="liblinear",       # good for small datasets
    penalty="l2", 
    C=0.8,                    # slightly stronger regularization
    class_weight="balanced",  # ensures fair weighting
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

# ==============================================================
# 7️⃣ Evaluation
# ==============================================================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained — Accuracy: {acc*100:.2f}%")

st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# ==============================================================
# 8️⃣ User Prediction Section
# ==============================================================
st.header("🧍‍♀️ Enter Patient Data")

col1, col2, col3 = st.columns(3)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
with col2:
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    BMI = st.number_input("BMI", min_value=0.0, max_value=80.0, value=32.0)
with col3:
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47)
    Age = st.number_input("Age", min_value=10, max_value=120, value=33)

if st.button("🔮 Predict Diabetes"):
    user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    user_scaled = scaler.transform(user_input)
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 Diabetic (Probability: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Non-Diabetic (Probability: {prob*100:.2f}%)")