import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="RAG Customer Support Assistant",
    page_icon="🤖"
)

st.title("🤖 RAG Customer Support Assistant")

st.write(
    "Ask questions about company policies, FAQs, and support documents."
)

# Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    ".",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

import os

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)

query = st.text_input(
    "Ask your question"
)

if query:

    with st.spinner("Searching documents..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a helpful customer support assistant.

Answer only from the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    with st.expander("Retrieved Context"):
        st.write(context)
