import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model_v2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_v2.pkl', 'rb'))

st.set_page_config(page_title="Research Evaluator", page_icon="🔬")

st.title("🔬 Research Paper Quality Evaluator")

st.write("Enter paper details below:")

# Input fields
title = st.text_input("Paper Title")
abstract = st.text_area("Paper Abstract", height=200)

# Prediction function
def predict_quality(title, abstract):
    text = title + " " + abstract
    
    length = len(abstract) / 1000
    word_count = len(abstract.split())
    unique_words = len(set(abstract.split()))
    keyword_score = sum(1 for k in ['method','model','analysis','dataset','results','experiment'] if k in abstract.lower())

    text_vec = vectorizer.transform([text]).toarray()

    features = np.hstack((
        text_vec,
        [[length, word_count, unique_words, keyword_score]]
    ))

    return model.predict(features)[0]

# Button
if st.button("Evaluate Quality"):
    if title and abstract:
        result = predict_quality(title, abstract)

        st.subheader("Prediction Result:")

        if result == "High":
            st.success("High Quality Paper ✅")
        elif result == "Medium":
            st.warning("Medium Quality Paper ⚠️")
        else:
            st.error("Low Quality Paper ❌")
    else:
        st.warning("Please enter both title and abstract")