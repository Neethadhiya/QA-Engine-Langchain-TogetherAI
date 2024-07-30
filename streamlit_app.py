import os
import streamlit as st
from dotenv import load_dotenv
from together import Together
import PyPDF2
import pandas as pd
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together.embeddings import TogetherEmbeddings

load_dotenv()

st.title("QA Engine 🐦")
st.subheader("🚀 Generate QA from any document")

# Initialize Together client
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

uploaded_file = st.file_uploader("Upload a file (CSV, PDF)", type=["csv", "pdf"])
docs = []

if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            csv_text = df.to_string()
            docs = [Document(page_content=csv_text)]
        
        elif uploaded_file.type == "application/pdf":
            text = ""
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
            docs = [Document(page_content=text)]
        
        st.success("File uploaded and processed successfully.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Input fields
ask_question = st.text_input("Ask a question about the document")

if docs:
    try:
        # Create and split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval"))

        # Set up the retriever
        retriever = vectorstore.as_retriever()

        # Define the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=client,
            retriever=retriever,
            chain_type="stuff",
            output_parser=StrOutputParser()
        )

        if st.button("Generate"):
            # Create the prompt
            question = ask_question

            if question:
                # Generate response
                answer = qa_chain.run(question)

                # Display the response content
                st.write(answer)
            else:
                st.warning("Please enter a question.")
    
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.info("Please upload a CSV or PDF file to proceed.")
