import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
data = {
    "resume": [
        "Python machine learning data analysis",
        "Java Spring Boot backend developer",
        "Deep learning NLP AI projects",
        "Accountant finance taxation"
    ],
    "label": ["AI", "Backend", "AI", "Finance"]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

st.title("AI Resume Classifier")

user_input = st.text_area("Paste Resume Text")

if st.button("Predict Role"):
    text_vector = vectorizer.transform([user_input])
    prediction = model.predict(text_vector)[0]
    st.success(f"Predicted Role: {prediction}")
