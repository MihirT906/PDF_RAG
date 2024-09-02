import re
from langchain.prompts import ChatPromptTemplate,  MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage

def clean_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-ASCII characters (optional)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Additional cleaning rules can be added here
    return text.strip()

def retrieve_relevant_docs(query, retriever):
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def create_rag_chain(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
    "You are an AI assistant for question-answering tasks. Use "
    "the following pieces of retrieved context from a flight regulation document to answer the "
    "question. Please return the most relevant information to the following question"
    "\n\n"
    "{context}"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
