import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Function definitions
def load_model_and_dataset(uploaded_file):
    dataset = pd.read_excel(uploaded_file)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = encode_questions(dataset['dialogue'].tolist(), model)
    return dataset, model, question_embeddings

def encode_questions(questions, model):
    return model.encode(questions, convert_to_tensor=True)

def get_answer(user_question, question_embeddings, dataset, model):
    user_question_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(user_question_embedding, question_embeddings)
    most_similar_idx = torch.argmax(cos_sim).item()
    return dataset.iloc[most_similar_idx]['summary ']

# Streamlit interface
st.title("Institute Admissions Chatbot")
st.write("Ask me anything about admissions!")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (Excel file)", type=["xlsx", "xls"])

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = {}

# Load dataset and model once file is uploaded
if uploaded_file is not None:
    if 'model' not in st.session_state or 'dataset' not in st.session_state:
        st.session_state.dataset, st.session_state.model, st.session_state.question_embeddings = load_model_and_dataset(uploaded_file)

# Interactive question and answer section
if 'dataset' in st.session_state:
    for i in range(len(st.session_state.questions) + 1):
        # Create a new question input
        user_question = st.text_input(f"Your question {i+1}:", key=f"question_{i+1}")
        
        # If there's a question, show the answer directly below it
        if user_question:
            answer = get_answer(user_question, st.session_state.question_embeddings, st.session_state.dataset, st.session_state.model)
            st.write(f"Answer {i+1}: {answer}")
            st.session_state.questions[f"question_{i+1}"] = user_question

# Button to add a new question input
if st.button("Ask us More!"):
    # Add a new key to the questions dictionary
    st.session_state.questions[f"question_{len(st.session_state.questions) + 1}"] = ""
