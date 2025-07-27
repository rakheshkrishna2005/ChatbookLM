import os
import gc
import ssl
import time
import socket
import atexit
import cohere
import tempfile
import warnings
import streamlit as st
import google.generativeai as genai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from weaviate import connect_to_weaviate_cloud
from weaviate.auth import Auth
from weaviate.collections.classes.config import Property, DataType, _Vectorizer
from weaviate.collections import Collection
from weaviate.classes.query import Filter, MetadataQuery

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

co = cohere.Client(COHERE_API_KEY)

if 'cohere_calls' not in st.session_state:
    st.session_state.cohere_calls = []
COHERE_RATE_LIMIT = 100
RATE_LIMIT_WINDOW = 60

def check_rate_limit():
    current_time = datetime.now()
    st.session_state.cohere_calls = [
        call_time for call_time in st.session_state.cohere_calls 
        if current_time - call_time < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]
    
    if len(st.session_state.cohere_calls) >= COHERE_RATE_LIMIT:
        oldest_call = min(st.session_state.cohere_calls)
        wait_time = RATE_LIMIT_WINDOW - (current_time - oldest_call).seconds
        if wait_time > 0:
            with st.sidebar:
                with st.spinner(f"⏳ Rate limit reached. Waiting {wait_time} seconds..."):
                    time.sleep(wait_time)
            st.session_state.cohere_calls = []
    
    st.session_state.cohere_calls.append(current_time)

def rate_limited_embed(texts, model, input_type):
    check_rate_limit()
    return co.embed(texts=texts, model=model, input_type=input_type)

client = None

def get_client():
    global client
    try:
        if client is None:
            cleanup_client()
            client = connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
            )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {str(e)}")
        cleanup_client()
        return None

def cleanup_client():
    global client
    try:
        if client:
            client.close()
            for conn in ssl.SSLSocket._client_sockets:
                try:
                    conn.close()
                except:
                    pass
            ssl.SSLSocket._client_sockets.clear()
            client = None
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")

def safe_cleanup():
    cleanup_client()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    
    try:
        for obj in gc.get_objects():
            if isinstance(obj, socket.socket):
                try:
                    obj.close()
                except:
                    pass
    except:
        pass

COLLECTION_NAME = "DocumentChunks"

def get_collection():
    client = get_client()
    if COLLECTION_NAME not in client.collections.list_all():
        client.collections.create(
            name=COLLECTION_NAME,
            vector_config=_Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ]
        )
    return client.collections.get(COLLECTION_NAME)

def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = file.name
            docs.extend(loaded)
            st.success(f"✅ Loaded {file.name}")
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    return docs

def index_documents(docs):
    collection = get_collection()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        embedding = rate_limited_embed(
            texts=[chunk.page_content],
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings[0]
        collection.data.insert(
            properties={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
            },
            vector=embedding
        )

def query_vector_db(query, selected_sources):
    collection = get_collection()
    query_vec = rate_limited_embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    filters = None
    if selected_sources:
        filters = Filter.by_property("source").contains_any(selected_sources)

    results = collection.query.near_vector(
        near_vector=query_vec,
        filters=filters,
        limit=5,
        return_metadata=MetadataQuery(distance=True),
    )

    return [obj.properties["text"] for obj in results.objects]

def generate_answer(query, context_chunks):
    prompt = (
        "Answer the question using the following context:\n\n"
        + "\n\n".join(context_chunks)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    response = model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="Chatbook LM - RAG", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    "<h1 style='text-align: center;'>📰 Chatbook LM</h1>",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🗃️ Source Selection")
    
    try:
        collection = get_collection()
        all_sources = list({obj.properties["source"] for obj in collection.query.fetch_objects(limit=100).objects})
        selected_sources = st.multiselect("Select Sources", options=all_sources, placeholder="Choose sources...", label_visibility="collapsed")
    except:
        selected_sources = []
    
    st.divider()
    
    st.header("📁 Source Management")
    uploaded_files = st.file_uploader(
        "Upload Sources",
        type=["pdf"],
        accept_multiple_files=True,
        key="file_uploader"
    )
        
    if st.button("📥 Index Sources", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload sources.")
        else:
            with st.spinner("📥 Indexing sources..."):
                try:
                    docs = load_documents(uploaded_files)
                    if docs:
                        index_documents(docs)
                        st.success(f"✅ {len(docs)} sources indexed!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No sources loaded.")
                except Exception as e:
                    st.error(f"⚠️ Indexing error! \n{str(e)}")
                    safe_cleanup()
    
    if st.button("🗑️ Delete Sources", use_container_width=True, type="secondary"):
        try:
            collection = get_collection()
            all_objects = collection.query.fetch_objects(limit=1000).objects
            if all_objects:
                with st.spinner("🗑️ Deleting sources..."):
                    for obj in all_objects:
                        collection.data.delete_by_id(obj.uuid)
                st.success(f"✅ {len(all_objects)} sources deleted!")
                st.session_state.messages = []
                st.rerun()
            else:
                st.info("ℹ️ No sources to delete.")
        except Exception as e:
            st.error(f"⚠️ Deletion error! \n{str(e)}")
            safe_cleanup()
    
    st.divider()
    
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if not selected_sources:
        st.warning("⚠️ Please select at least one source from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    relevant_chunks = query_vector_db(prompt, selected_sources)
                    if relevant_chunks:
                        answer = generate_answer(prompt, relevant_chunks)
                        st.markdown(answer)
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # with st.expander("📄 View Context Sources"):
                        #     for i, chunk in enumerate(relevant_chunks, 1):
                        #         st.markdown(f"**Source {i}**:\n{chunk[:200]}...")
                    else:
                        no_answer = "I couldn't find relevant information in the selected sources. Please try rephrasing your question or select different sources."
                        st.markdown(no_answer)
                        st.session_state.messages.append({"role": "assistant", "content": no_answer})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    safe_cleanup()

atexit.register(safe_cleanup)