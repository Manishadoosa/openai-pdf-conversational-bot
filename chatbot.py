import streamlit as st
import PyPDF2
from transformers import pipeline

# Function to extract text from PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to answer user questions
def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app layout
st.title('PDF Conversational Chatbot')

# Upload PDF file
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    # Extract text from PDF file
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Display chat conversation
    form_key = "chat_form"  # Unique key for the form
    with st.form(form_key):
        user_input = st.text_input("You:")
        if st.form_submit_button("Send"):
            # Process user input and generate response
            if "what the pdf about" in user_input.lower():
                # Answer the question about the PDF content
                answer = answer_question("What is the PDF about?", pdf_text)
                st.write("Bot:", answer)
            else:
                # Answer other user questions if applicable
                answer = answer_question(user_input, pdf_text)
                st.write("Bot:", answer)
