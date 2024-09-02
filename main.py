import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
import re
import PyPDF2

# Custom utility functions
from utils import clean_text, retrieve_relevant_docs, create_rag_chain

# Set up environment variables
groq_api_key=st.secrets['GROQ_API_KEY']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Streamlit application title
st.title("AI Assistant for understanding flight regulations")

# Specify the path to the local PDF file
pdf_path = 'flight-regulations.pdf'  

if os.path.exists(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    # Clean and prepare document text
    doc_text = clean_text(text)
    single_document = Document(page_content=doc_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150) 
    docs = text_splitter.split_documents([single_document])

    # Create embeddings
    embedding_model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create vector store and retriever
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3})

    # Set up the conversational AI model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0)
    
    # Create conversational RAG Chain
    rag_chain = create_rag_chain(llm, retriever)

    def continual_chat():
        st.write("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history = []  # Collect chat history here (a sequence of messages)
        

        sample_questions = [
            "",  # Empty default option
            "Time difference between Singapore and India is 2 and a half hours. If a crew member flies from India to Singapore, do they need to be acclimatised?",
            "Can you show all rules that are related to consecutive night?",
            "Is it okay for a crew member to fly 80 hours in a 14 day consecutive window?"
        ]
        
        query = st.radio("Choose a sample question or type your own:", sample_questions)
        
        # Text input for custom query
        custom_query = st.text_input("Or type your question here:")
        
        # Use the custom query if provided, otherwise use the selected sample question
        if custom_query:
            query = custom_query
            
        if query:
            if query.lower() == "exit":
                st.write("Conversation ended.")
                return
            
            # Process the user's query through the retrieval chain
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            # Display the AI's response
            st.write(f"AI: {result['answer']}")
            
            # Display relevant documents
            st.write("Relevant Documents:")
            relevant_docs = retrieve_relevant_docs(query, retriever)
            for i, doc in enumerate(relevant_docs, 1):
                st.write(f"Source {i}:")
                st.write(doc.page_content)
            
            # Update the chat history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=result["answer"]))
    
    continual_chat()
else:
    st.error(f"The specified PDF file at {pdf_path} does not exist.")
