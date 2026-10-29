import os
import time
import speech_recognition as sr
import sounddevice as sd
from scipy.io import wavfile
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import pyttsx3  # Added for Text-to-Speech conversion

# 1. Capture Voice via sounddevice and Convert to Text
def record_and_transcribe():
    sample_rate = 16000  # standard for speech recognition
    duration = 5         # record for 5 seconds
    temp_filename = "temp_voice.wav"
    
    print("\n[🎙️ Listening...] Speak your question clearly into your microphone now...")
    
    try:
        # Record audio data natively from hardware microphone
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until the 5-second recording window finishes
        print("[⚙️ Processing Voice...]")
        
        # Save audio payload to a temporary file
        wavfile.write(temp_filename, sample_rate, audio_data)
        
        # Feed the temporary wave file into SpeechReader
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio = recognizer.record(source)
            
        text_query = recognizer.recognize_google(audio)
        print(f"👉 You Said: \"{text_query}\"")
        
        # Clean up the file after use
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return text_query
        
    except sr.UnknownValueError:
        print("❌ Error: Could not cleanly understand the audio format.")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None
    except Exception as e:
        print(f"❌ Microphone Stream Error: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return None

# 2. Lightweight Document Loader
def load_local_documents(docs_dir):
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
                print(f"Warning: Couldn't read {filename} -> {e}")
    return documents

# 3. Text-to-Speech Output Function
def speak_text(text):
    """Initializes TTS engine dynamically and speaks the given text string."""
    try:
        engine = pyttsx3.init()
        # Optional tuning (uncomment to adjust speed or volume)
        # engine.setProperty('rate', 150)    # Speed percent (lower is slower)
        # engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"⚠️ Text-to-Speech Error: {e}")

# 4. Main System Flow
def main():
    docs_dir = "./docs"
    docs = load_local_documents(docs_dir)
    
    if not docs:
        print(f"\n⚠️ The directory '{docs_dir}' is empty! Drop a '.txt' file with some text inside it first.")
        input("\nPress Enter to exit...")
        return

    print(f"📚 Successfully loaded {len(docs)} document source(s). Creating local vector map...")
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Initialize your local open-source LLM
    llm = ChatOllama(model="llama3.2", temperature=0.3)

    # Structure how your AI behaves
    prompt = ChatPromptTemplate.from_template(
        "You are a professional assistant answering questions based strictly on the following text context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # Execution Chain
    chain = prompt | llm | StrOutputParser()

    # Trigger Audio Interface
    user_query = record_and_transcribe()
    
    if user_query:
        print("\n🧠 Querying local LLM against your text files...")
        response = chain.invoke({"context": context_text, "question": user_query})
        
        print("\n🤖 AI Answer:")
        print(response)
        
        # New block: Convert the generated LLM text answer to audio output
        print("\n🔊 Speaking response...")
        speak_text(response)
        
    print("\n" + "="*40)
    input("Execution finished. Press [Enter] to cleanly close the session...")

if __name__ == "__main__":
    main()
