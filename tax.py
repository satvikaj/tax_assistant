
import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
    .stChatInput {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Configure Google Gemini
genai.configure(api_key="AIzaSyALSp9jVTV04ziisP5IdFu5-VpPsx39NfU")  # Replace with your Gemini API key
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

# Load data and create FAISS index
@st.cache_data
def load_data():
    try:
        # Replace this with your dataset
        data = {
            "question": [
                "What are the income tax slabs under the new tax regime for FY 2025-26 (AY 2026-27)?",
                "What are the income tax slabs under the new tax regime for FY 2024-25 (AY 2025-26)?",
                "What are the changes in the new tax regime in Budget 2025?",
                "What is the highest tax rate under the new tax regime?",
                "What is the highest surcharge rate in the new tax regime?",
                "What is the standard deduction under the new tax regime?",
                "What are the tax slabs under the old tax regime for individuals below 60 years?",
                "What are the tax slabs under the old tax regime for senior citizens (60-80 years)?",
                "What are the tax slabs under the old tax regime for super senior citizens (80+ years)?",
                "What is the tax rebate under Section 87A for FY 2025-26?",
                "What is the standard deduction for employer's NPS contribution?",
                "Is the new tax regime mandatory?",
                "How does one choose between the new and old tax regimes?",
            ],
            "answer": [
                "Rs 0- Rs 4 lakh: Nil, Rs 4 lakh - Rs 8 lakh: 5%, Rs 8 lakh - Rs 12 lakh: 10%, Rs 12 lakh - Rs 16 lakh: 15%, Rs 16 lakh - Rs 20 lakh: 20%, Rs 20 lakh - Rs 24 lakh: 25%, Above Rs 24 lakh: 30%",
                "Rs 0 - Rs 3 lakh: Nil, Rs 3 lakh - Rs 7 lakh: 5%, Rs 7 lakh - Rs 10 lakh: 10%, Rs 10 lakh - Rs 12 lakh: 15%, Rs 12 lakh - Rs 15 lakh: 20%, Above Rs 15 lakh: 30%",
                "The basic exemption limit increased to Rs 4 lakh from Rs 3 lakh. Tax rebate under Section 87A now applies to taxable incomes up to Rs 12 lakh, making tax payable zero for income up to this limit.",
                "30% for income above Rs 24 lakh.",
                "25% for those earning above Rs 2 crore.",
                "Rs 75,000 for salaried individuals and Rs 25,000 for family pensioners.",
                "Rs 0 - Rs 2.5 lakh: Nil, Rs 2.5 lakh - Rs 5 lakh: 5%, Rs 5 lakh - Rs 10 lakh: 20%, Above Rs 10 lakh: 30%",
                "Rs 0 - Rs 3 lakh: Nil, Rs 3 lakh - Rs 5 lakh: 5%, Rs 5 lakh - Rs 10 lakh: 20%, Above Rs 10 lakh: 30%",
                "Rs 0 - Rs 5 lakh: Nil, Rs 5 lakh - Rs 10 lakh: 20%, Above Rs 10 lakh: 30%",
                "Applicable for taxable incomes up to Rs 12 lakh, making tax payable zero up to this limit.",
                "14% from FY 2025-26, previously 10%.",
                "No, the new tax regime is the default, but individuals can opt for the old tax regime.",
                "Individuals with deductions and exemptions (e.g., HRA, 80C, 80D) may find the old tax regime beneficial, while those with fewer deductions may prefer the new regime.",
            ],
        }
        df = pd.DataFrame(data)
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load dataset and FAISS index
df, faiss_index = load_data()

# App Header
st.markdown('<h1 class="chat-font">🤖 Income Tax Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything about income tax slabs and regimes!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']  # Return the closest answer
    return None

# Function to generate a better answer using Gemini
def generate_better_answer(query, retrieved_answer):
    prompt = f"""You are an income tax expert. Provide a detailed and user-friendly response to the following question:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Explain the answer in simple terms.
    - Add context or examples if necessary.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Function to generate a response for out-of-scope questions using Gemini
def generate_out_of_scope_response():
    prompt = """You are an income tax expert. Respond to the following question in a friendly and conversational tone:
    - If the question is outside your knowledge, respond with: "I'm sorry, I cannot answer that question."
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about income tax..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Find the closest answer
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                # Generate a better answer using Gemini
                better_answer = generate_better_answer(prompt, retrieved_answer)
                response = f"**Income Tax Expert**:\n{better_answer}"
            else:
                # Use Gemini for out-of-scope questions
                response = generate_out_of_scope_response()
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
