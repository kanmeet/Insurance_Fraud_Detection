import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

st.title("🏥 Insurance Fraud Detection")
st.write("Using Machine Learning to predict fraudulent insurance claims.")
st.info("Please fill out the form on the left and click **Predict Fraud** to classify a new claim.")

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mehtabhavin10/insurance_fraud_detection/master/dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Drop irrelevant or high-cardinality columns
df.drop(['policy_number', 'policy_bind_date', 'incident_date', 'incident_location',
         'auto_make', 'auto_model', 'auto_year'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split into features and target
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cache model training
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X_train, y_train)

# Sidebar form for input
st.sidebar.header("Enter Claim Details")
user_input = {}
for col in X.columns:
    if len(df[col].unique()) < 10:
        user_input[col] = st.sidebar.selectbox(f"{col}", sorted(df[col].unique()))
    else:
        user_input[col] = st.sidebar.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Predict
if st.sidebar.button("Predict Fraud"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 This claim is predicted to be **Fraudulent**.")
    else:
        st.success("✅ This claim is predicted to be **Genuine**.")

# Evaluation metrics
st.subheader("📊 Model Evaluation on Test Set")
y_pred = model.predict(X_test)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion matrix
st.write("Confusion Matrix:")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Feature importances
st.subheader("🔍 Feature Importances")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x=feat_imp, y=feat_imp.index, ax=ax2)
st.pyplot(fig2)
