import streamlit as st
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import io

# 1. Page Configuration & Title
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="centered")
st.title("🎙️ Web-Ready Voice Assistant")
st.write("Click the button below, speak your prompt, and stop recording to submit.")

# 2. Initialize the Cloud AI Engine (Bypasses Local Ollama Dependency)
# It will look for the API key in your Streamlit Cloud Settings secrets manager
try:
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.7,
    )
except Exception as e:
    st.error("API Key missing. Please configure GROQ_API_KEY in the Streamlit Advanced Settings.")

# 3. Render Browser Microphone Component
audio_data = mic_recorder(
    start_prompt="🎵 Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    key="web_mic"
)

# 4. Process Audio when recording finishes
if audio_data and 'bytes' in audio_data:
    st.audio(audio_data['bytes'], format="audio/wav")
    
    with st.spinner("Processing your voice input..."):
        # For a production app, pass audio_data['bytes'] to an audio transcription API like Whisper here.
        # For this execution frame, we mock the text extraction to proceed directly to the cloud LLM pipeline.
        mocked_transcription = "Hello! Tell me a short story about an AI learning to speak."
        
        st.info(f"🗣️ **Transcribed Text:** {mocked_transcription}")
        
    with st.spinner("Generating AI Response..."):
        try:
            response = llm.invoke([HumanMessage(content=mocked_transcription)])
            
            st.success("🤖 **AI Assistant Response:**")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error communicating with AI engine: {e}")
