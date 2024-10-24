import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Streamlit interface
st.title("Institute Admissions Chatbot")
st.write("Ask me anything about admissions!")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx", "xls"])

# Check if file is uploaded
if uploaded_file is not None:
    # Load the dataset
    dataset = pd.read_excel(uploaded_file)

    # Initialize the SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Function to encode questions into embeddings
    def encode_questions(questions):
        return model.encode(questions, convert_to_tensor=True)

    # Encode all questions in the dataset
    question_embeddings = encode_questions(dataset['dialogue'].tolist())

    # Function to find the most similar question and return its answer
    def get_answer(user_question):
        user_question_embedding = model.encode(user_question, convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(user_question_embedding, question_embeddings)
        most_similar_idx = torch.argmax(cos_sim).item()
        return dataset.iloc[most_similar_idx]['summary ']

    # User input
    user_question = st.text_input("Your question:")

    if user_question:
        answer = get_answer(user_question)
        st.text("Answer: ")
        st.write(answer)
