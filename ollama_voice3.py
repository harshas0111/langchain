import streamlit as st
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Page Configuration & Title
st.set_page_config(page_title="Tennis Rules AI Assistant", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Rules Voice Assistant")
st.write("Click the button, ask a question about tennis rules, and stop recording to submit.")

# 2. Initialize the Cloud AI Engine with Free-Tier Safety Limits
try:
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2,   # Low temperature keeps the answers highly factual
        max_tokens=200,    # Fixes the 429 Output Tokens Per Minute (OTPM) rate limit error
    )
except Exception as e:
    st.error("API Key missing. Please configure GROQ_API_KEY in the Streamlit Advanced Settings.")

# 3. Inject the Tennis Rules Document into the AI's System Memory
system_prompt = """
You are a precise Tennis Rules Expert. Answer the user's question accurately using ONLY the rules provided below. 
If the answer cannot be found in these rules, politely tell the user that you don't have that specific detail.

TENNIS RULES CONTEXT:
1. OBJECTIVE: Played as singles (1v1, 27ft wide) or doubles (2v2, 36ft wide). Hit ball over net into opponent's court.
2. SERVING: Server stands behind baseline, hits diagonally into opposite service box. Alternates right/left sides after each point. Two attempts allowed. Double fault loses point. Let (clips net, lands in box) is replayed. Ace is untouched legal serve. Server's score called first.
3. SCORING: 0 = Love, 1 = 15, 2 = 30, 3 = 40. Need 4 points and a 2-point lead to win. 40-40 is Deuce. Advantage scoring requires 2 consecutive points from deuce to win (Ad-in: server advantage; Ad-out: receiver advantage). Some matches use no-ad scoring.
4. MATCH STRUCTURE: Set won by first to 6 games with a 2-game lead. Tiebreak at 6-6. Best of 3 sets typically (best of 5 in some men's Grand Slams). Change ends after odd-numbered games (1, 3, 5).
5. RALLIES & LOSING POINTS: Ball can bounce at most once. Lines count as IN. You lose the point if the ball bounces twice, goes out of bounds, you hit it twice, touch the net during a point, or hit it before it crosses the net. Ball touching the net during a rally is legal if it lands inside.
6. SHOTS: Forehand (dominant side), Backhand (across body), Drop shot (soft over net), Smash (forceful overhead), Slice (backspin, stays low).
"""

# 4. Render Browser Microphone Component
audio_data = mic_recorder(
    start_prompt="🎵 Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    key="tennis_mic"
)

# 5. Process Input when recording finishes
if audio_data and 'bytes' in audio_data:
    st.audio(audio_data['bytes'], format="audio/wav")
    
    with st.spinner("Processing your voice input..."):
        # Simulated voice transcription pipeline for the execution frame
        mocked_question = "What happens if a serve hits the net line but falls in the right box?"
        
        st.info(f"🗣️ **Your Question:** {mocked_question}")
        
    with st.spinner("Analyzing rules..."):
        try:
            # Package the structural prompt context alongside the query
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=mocked_question)
            ]
            
            response = llm.invoke(messages)
            
            st.success("🤖 **Rules Expert Verdict:**")
            st.write(response.content)
        except Exception as e:
            st.error(f"Error communicating with AI engine: {e}")
