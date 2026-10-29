import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import speech_recognition as sr
import pyttsx3

# --- WINDOWS 11 STYLING & CONFIG ---
st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Voice-Enabled AI Document Assistant")
st.caption("Talk to your local knowledge base using LangChain, Ollama (Llama 3.2), and Voice Synthesis.")

# --- DOCUMENT LOADER ---
def load_local_documents(docs_dir="./docs"):
    documents = []
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        return documents
        
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
                    if text_content.strip():
                        documents.append(Document(page_content=text_content, metadata={"source": filename}))
            except Exception as e:
                st.sidebar.error(f"Error reading {filename}: {e}")
    return documents

# --- TEXT TO SPEECH (WINDOWS NATIVE) ---
def speak_text(text):
    """Speaks the response text using the offline pyttsx3 engine."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.sidebar.warning(f"Audio playback issue: {e}")

# --- SYSTEM INITIALIZATION ---
docs_dir = "./docs"
docs = load_local_documents(docs_dir)

# Sidebar Knowledge Base Context
st.sidebar.header("📚 Knowledge Base")
if docs:
    st.sidebar.success(f"Loaded {len(docs)} document(s) successfully.")
    with st.sidebar.expander("View Active Files"):
        for doc in docs:
            st.sidebar.write(f"- {doc.metadata['source']}")
else:
    st.sidebar.error(f"The '{docs_dir}' directory is empty!")
    st.sidebar.info("Drop '.txt' files into the `./docs` folder to provide context.")

# Initialize the RAG Chain if context exists
if docs:
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Initialize Ollama
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a professional assistant answering questions based strictly on the following text context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()

    # --- PRESENTATION INTERFACE ---
    st.write("### 🎛️ Hybrid Input Dashboard")
    
    # Native Streamlit audio recorder (No PyAudio required!)
    audio_file = st.audio_input("🎤 Record your question:")
    
    # Client Fallback Textbox
    text_query = st.text_input("💬 Or type your question here if the room is too loud:", placeholder="Ask something about the documents...")

    user_query = None

    # Priority 1: Handle text input if typed
    if text_query:
        user_query = text_query
        st.chat_message("user").write(f"**You typed:** {user_query}")

    # Priority 2: Handle voice input if recorded
    elif audio_file is not None:
        temp_audio_path = "temp_windows_voice.wav"
        
        # Read the raw browser bytes and save safely to local workspace
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
            
        recognizer = sr.Recognizer()
        with st.spinner("🧠 Transcribing voice input..."):
            try:
                with sr.AudioFile(temp_audio_path) as source:
                    audio_data = recognizer.record(source)
                user_query = recognizer.recognize_google(audio_data)
                st.chat_message("user").write(f"**You said:** {user_query}")
            except sr.UnknownValueError:
                st.error("❌ Audio could not be understood. Try speaking closer to the microphone or use the text box.")
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    # Process whichever input was captured
    if user_query:
        with st.spinner("🤖 Generating answer from local knowledge base..."):
            response = chain.invoke({"context": context_text, "question": user_query})
        
        st.chat_message("assistant").write(response)
        
        with st.spinner("🔊 Speaking response out loud..."):
            speak_text(response)
else:
    st.warning("⚠️ Please populate the `./docs` directory to activate the voice dashboard system.")
