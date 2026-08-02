import streamlit as st  # type: ignore
import os
from PyPDF2 import PdfReader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_core.runnables import RunnablePassthrough  # type: ignore
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file (see .env.example)
load_dotenv()

# Prefer a key from the environment; fall back to letting the user paste one in.
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

# Current, non-preview model names (langchain-google-genai 4.x)
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            with st.spinner(f"Reading {pdf.name}..."):
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"PDF read error: {str(e)}")
        st.error(f"Failed to read PDF: {str(e)}")
        return None


def get_text_chunks(text):
    try:
        if not text:
            raise ValueError("Empty text content")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300
        )
        return text_splitter.split_text(text)
    except Exception as e:
        logging.error(f"Text split error: {str(e)}")
        st.error(f"Document processing failed: {str(e)}")
        return None


def get_vector_store(text_chunks, api_key):
    try:
        if not text_chunks:
            raise ValueError("No text chunks to process")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key
        )

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        try:
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:
            logging.error(f"Index save error: {str(e)}")
            st.error("Failed to save document index")
            return False

    except Exception as e:
        logging.error(f"Embedding error: {str(e)}")
        st.error("Failed to create document embeddings")
        return False


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(retriever, api_key):
    """Build a simple retrieval-augmented generation chain with LCEL."""
    try:
        prompt_template = """
        Answer based on the provided context. If the query is unrelated to the context, provide a general response from external knowledge sources.

        Context: {context}

        Question: {input}

        Answer:"""

        model = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=0.3,
            google_api_key=api_key
        )

        prompt = PromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return rag_chain

    except Exception as e:
        logging.error(f"Model init error: {str(e)}")
        st.error("AI model initialization failed")
        return None


def user_input(user_question, api_key):
    try:
        if not user_question.strip():
            raise ValueError("Empty question")

        if not os.path.exists("faiss_index"):
            st.error("Please process PDFs first!")
            return None

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=api_key
            )
            vector_store = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Index load error: {str(e)}")
            st.error("Failed to load document index")
            return None

        retriever = vector_store.as_retriever()
        rag_chain = get_rag_chain(retriever, api_key)

        if not rag_chain:
            return None

        try:
            answer = rag_chain.invoke(user_question)
            return answer or "No answer found"
        except Exception as e:
            logging.error(f"Query error: {str(e)}")
            return "Error generating answer"

    except Exception as e:
        logging.error(f"Input handling error: {str(e)}")
        return f"Processing error: {str(e)}"


def main():
    try:
        st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
        st.header("Chat with Multiple PDFs 📚")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        with st.sidebar:
            try:
                st.subheader("Google API Key")
                if os.getenv("GOOGLE_API_KEY"):
                    st.caption("Using GOOGLE_API_KEY from environment.")
                else:
                    st.session_state.google_api_key = st.text_input(
                        "Enter your Google API key",
                        value=st.session_state.google_api_key,
                        type="password",
                        help="Not set in the environment, so paste it here. "
                             "It's kept only in this browser session, not saved to disk."
                    )

                st.subheader("Document Management")
                pdf_docs = st.file_uploader(
                    "Upload PDF files here",
                    accept_multiple_files=True,
                    type="pdf"
                )

                if st.button("Process PDFs", disabled=not pdf_docs or not st.session_state.google_api_key):
                    with st.spinner("Processing documents..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks and get_vector_store(text_chunks, st.session_state.google_api_key):
                                st.success("Processing complete! Ask questions now")

                if not st.session_state.google_api_key:
                    st.info("Enter a Google API key above to enable processing.")

                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

            except Exception as e:
                logging.error(f"Sidebar error: {str(e)}")
                st.error("Sidebar operation failed")

        # Display chat history at the top
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for chat in st.session_state.chat_history:  # Natural order (oldest first)
                with st.chat_message("user"):
                    st.markdown(chat["question"])
                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])
                st.divider()

        # Chat input at bottom with latest messages appearing above it
        user_question = st.chat_input("Ask about your documents:")

        if user_question and not st.session_state.google_api_key:
            st.error("Please enter a Google API key in the sidebar first.")
        elif user_question:
            with st.spinner("Analyzing..."):
                answer = user_input(user_question, st.session_state.google_api_key)
                if answer:
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
            st.rerun()

    except Exception as e:
        logging.critical(f"App critical error: {str(e)}")
        st.error("Application encountered a critical error. Please refresh the page.")


if __name__ == "__main__":
    main()
