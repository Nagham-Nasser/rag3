import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
os.environ["STREAMLIT_WATCH_FILE"] = "none"  # Disables Streamlit file watching entirely for compatibility with PyTorch

# Disable Streamlit's file watcher for PyTorch compatibility
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

@st.cache_data
def load_and_process_pdf():
    try:
        loader = PyPDFLoader("yolov9_paper.pdf")
        data = loader.load()
        from langchain_core.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        return docs
    except Exception as e:
        st.error(f"Error loading or processing PDF: {e}")
        st.stop()

@st.cache_resource
def create_embeddings(_docs):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(_docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.stop()

docs = load_and_process_pdf()
vectorstore = create_embeddings(docs)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

hf_pipeline = pipeline("text-generation", model="gpt2")

llm = HuggingFacePipeline(pipeline=hf_pipeline)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("El Fayrouz")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['prompt']}")
    st.write(f"**Bot:** {entry['answer']}")
    st.write("---")

query = st.text_input("Enter your question:", key="query_input")

if st.button("Submit"):
    if query:
        try:
            with st.spinner("Generating response..."):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]

                st.session_state.chat_history.append({"prompt": query, "answer": answer})
                st.rerun()
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")
