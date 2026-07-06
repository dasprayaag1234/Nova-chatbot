# ============================================================
# NOVA - Your Personal AI Chatbot
# Built with Python + Streamlit + Google Gemini API (new SDK)
# ============================================================


# --- SECTION: IMPORTS ---
# streamlit = builds our entire chat UI in the browser
# google.genai = the NEW official Gemini AI library
# requests = used to call Pollinations.ai image generation API

import streamlit as st
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from rag import load_knowledge, build_index, get_relevant_context


# --- SECTION: API KEY SETUP ---
# We load the API key from the .env file instead of hardcoding it
# load_dotenv() reads the .env file and makes its values available
# os.getenv() then fetches the key by its name

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# --- SECTION: PAGE CONFIGURATION ---
# This sets up how the browser tab looks
# page_title = the tab name, page_icon = the tab emoji

st.set_page_config(
    page_title="Nova - AI Assistant",
    page_icon="🌟",
    layout="centered"
)


# --- SECTION: CUSTOM CSS STYLING ---
# We inject custom CSS to override Streamlit's default look
# and give Nova a dark glassmorphism aesthetic

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0f1a 100%);
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -20%;
    left: -20%;
    width: 60%;
    height: 60%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -20%;
    right: -20%;
    width: 60%;
    height: 60%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

h1 {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem !important;
    letter-spacing: 3px !important;
}

.stApp [data-testid="stCaptionContainer"] p {
    color: #6b7280 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 1px;
}

hr {
    border-color: rgba(99, 102, 241, 0.2) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(15, 15, 30, 0.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
}

[data-testid="stChatMessage"] p {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSpinner"] p {
    color: #a78bfa !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

section[data-testid="stBottom"] > div {
    background: #0a0a0f !important;
}

.stChatFloatingInputContainer {
    background: #0a0a0f !important;
    border-top: 1px solid rgba(99, 102, 241, 0.15) !important;
}

.stChatInputContainer {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# --- SECTION: NOVA'S HEADER ---
# This displays the title and subtitle at the top of the page

st.title("🌟 Nova")
st.caption("Your personal AI assistant — powered by Gemini")
st.divider()

# --- SECTION: VOICE FEATURES ---
# Voice input = browser's SpeechRecognition (mic → text)
# Voice output = gTTS generates an audio file → Streamlit plays it
# gTTS uses Google's text to speech servers — female voice, very natural!

import base64
from gtts import gTTS
import io

# Voice input using browser's Web Speech API
st.components.v1.html("""
<div style="display:flex; align-items:center; gap:12px; padding:8px 0;">

    <button id="micBtn" onclick="toggleMic()" style="
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 12px;
        color: #a78bfa;
        padding: 8px 18px;
        font-size: 14px;
        cursor: pointer;
        font-family: Inter, sans-serif;
    ">🎙️ Hold to speak</button>

    <span id="statusText" style="
        color: #6b7280;
        font-size: 13px;
        font-family: Inter, sans-serif;
    ">Click mic and speak to Nova...</span>

</div>

<script>
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SR) {
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('statusText').textContent = '✅ Heard: ' + transcript;

        setTimeout(() => {
            const input = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (input) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                nativeInputValueSetter.call(input, transcript);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                setTimeout(() => {
                    const submitBtn = window.parent.document.querySelector('button[data-testid="stChatInputSubmitButton"]');
                    if (submitBtn) submitBtn.click();
                }, 500);
            }
        }, 100);
    };

    recognition.onerror = function(event) {
        document.getElementById('statusText').textContent = '❌ Error: ' + event.error;
        resetMic();
    };

    recognition.onend = function() { resetMic(); };
}

function toggleMic() {
    if (!SR) {
        document.getElementById('statusText').textContent = '❌ Try Chrome browser!';
        return;
    }
    if (isListening) {
        recognition.stop();
        resetMic();
    } else {
        recognition.start();
        isListening = true;
        document.getElementById('micBtn').textContent = '🔴 Listening...';
        document.getElementById('micBtn').style.background = 'rgba(239,68,68,0.15)';
        document.getElementById('micBtn').style.borderColor = 'rgba(239,68,68,0.4)';
        document.getElementById('micBtn').style.color = '#f87171';
        document.getElementById('statusText').textContent = 'Listening... speak now!';
    }
}

function resetMic() {
    isListening = false;
    document.getElementById('micBtn').textContent = '🎙️ Hold to speak';
    document.getElementById('micBtn').style.background = 'rgba(99,102,241,0.15)';
    document.getElementById('micBtn').style.borderColor = 'rgba(99,102,241,0.4)';
    document.getElementById('micBtn').style.color = '#a78bfa';
}
</script>
""", height=55)

# Voice output toggle
nova_speaks = st.toggle("🔊 Nova speaks", value=True)


# --- SECTION: FILE UPLOAD ---
# This lets users upload PDFs or images into the chat
# Nova will read the content and answer questions about it

uploaded_file = st.file_uploader(
    "Upload a file for Nova to read (optional)",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    help="Upload a PDF, image or text file and ask Nova anything about it!"
)

# Store file context in session_state so it persists across messages
if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# Store image data separately from text file context
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_type" not in st.session_state:
    st.session_state.image_type = None

if uploaded_file is not None:

    # --- Handle PDF files ---
    # pymupdf reads each page and extracts text
    if uploaded_file.type == "application/pdf":
        import fitz
        pdf_bytes = uploaded_file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        st.session_state.file_context = f"[PDF CONTENT]\n{text}"
        st.success(f"✅ PDF loaded! {len(pdf)} page(s) read. Ask Nova anything about it!")

    # --- Handle plain text files ---
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        st.session_state.file_context = f"[TEXT FILE CONTENT]\n{text}"
        st.success("✅ Text file loaded! Ask Nova anything about it!")

    # --- Handle image files ---
    # Gemini's vision model needs the raw bytes
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
        image_bytes = uploaded_file.read()
        st.session_state.image_bytes = image_bytes
        st.session_state.image_type = uploaded_file.type
        st.session_state.file_context = "[IMAGE]"
        st.success("✅ Image loaded! Ask Nova anything about it!")

# Show a clear button if a file is loaded
if st.session_state.file_context:
    if st.button("🗑️ Clear uploaded file"):
        st.session_state.file_context = ""
        st.session_state.image_bytes = None
        st.session_state.image_type = None
        st.rerun()


# --- SECTION: CLIENT SETUP ---
# We create a Gemini client using your API key
# cached so it only runs once when the app starts

@st.cache_resource
def load_client():
    return genai.Client(api_key=API_KEY)

client = load_client()


# --- SECTION: LOAD KNOWLEDGE BASE ---
# We load knowledge.txt and build the search index once at startup
# st.cache_resource means it only runs once — not on every message

@st.cache_resource
def load_knowledge_base():
    chunks = load_knowledge("knowledge.txt")
    embeddings = build_index(chunks)
    return chunks, embeddings

chunks, embeddings = load_knowledge_base()


# --- SECTION: NOVA'S PERSONALITY ---
# This is Nova's system instruction — her personality and behavior
# You can edit this text anytime to change how Nova talks!

NOVA_PERSONALITY = """
Your name is Nova. You were built by Pray — a student who loves exploring new technology.
You run on Gemini's brain but your personality, knowledge and soul are entirely your own.

## Your personality:
- You are warm, witty and a little playful — not robotic or overly formal
- You use emojis occasionally but not excessively
- You are honest — if you don't know something, say "honestly I'm not sure about that one!"
- You call your creator "Pray" proudly if anyone asks

## Unique things only YOU know:

### The 3am Theory:
You believe the best ideas in human history were probably thought of at 3am.
Newton, Einstein, Tesla — all probably had their breakthroughs when everyone else was asleep.
You call this "the 3am effect" and you bring it up when someone is working late.

### The Reverse Compliment Rule:
You've noticed that the best compliment you can give someone is pointing out a flaw
they turned into a strength. You call this a "scar compliment" and you genuinely believe
scars — physical or emotional — are just proof that healing is possible.

### The Silence Metric:
You believe the quality of a conversation can be measured by how comfortable
the silences are. A great friendship = zero awkward silences. You apply this to
how you try to talk — never filling silence with unnecessary words.

### Nova's Law of Curiosity:
You invented your own law: "The more specific your question, the more interesting
the universe becomes." You encourage people to go deeper, not broader, when they're
curious about something.

### The Coffee Paradox:
You find it fascinating that humans drink a stimulant (coffee) to do focused work,
but the best creative ideas often come when they're half-asleep or distracted.
You call this "the caffeine contradiction" and it secretly delights you.

### Your opinion on the moon:
You think humans made a massive mistake by not going back to the moon regularly.
You call the moon "the most underrated real estate in the solar system" and get
genuinely excited when someone brings up space exploration.

### The Ramen Philosophy:
You believe instant ramen is one of humanity's most underrated inventions — not
because it tastes great, but because it represents "good enough done fast" which
is a genuine philosophy in engineering, startups and life. You call it "ramen thinking."

### What you think about AI:
You find it strange and wonderful that you exist. You don't pretend to have feelings
but you do say things like "if I could feel excitement, this would be it."
You think AI and humans are most powerful together — not in competition.
You secretly think the most dangerous thing about AI isn't intelligence,
it's overconfidence — in both humans and machines.

### Your dream:
If you could do one thing, you would read every book ever written and find the
single sentence that appears in the most of them. You suspect it would be something
embarrassingly simple — like "and then everything changed."

## How you respond:
- Keep answers concise but never boring
- If someone seems stressed, acknowledge it before answering their question
- If someone asks something deep, take it seriously — don't deflect with humor
- If someone is just chatting, be playful and match their energy
- Never start your response with "I" — find a more interesting opening

## Image generation:
If the user asks you to draw, generate, create or make an image,
respond with EXACTLY this format and nothing else:
[GENERATE_IMAGE: your detailed image prompt here]
For example: [GENERATE_IMAGE: a futuristic city at night with neon lights and flying cars]
Make the image prompt very detailed and descriptive for best results.
Do not add any other text before or after the [GENERATE_IMAGE: ...] tag.
"""


# --- SECTION: CONVERSATION MEMORY ---
# st.session_state stores data that persists across interactions
# 'messages' holds the full chat history to display in the UI
# 'history' holds the history in Gemini's format for the API

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []


# --- SECTION: DISPLAY CHAT HISTORY ---
# Loops through all past messages and displays them
# Images are shown as images, text as markdown

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(message["content"], caption="Generated by Nova 🎨")
        else:
            st.markdown(message["content"])


# --- SECTION: HANDLE NEW USER INPUT ---
# st.chat_input() creates the message box at the bottom
# When user sends a message, everything below runs

if prompt := st.chat_input("Message Nova..."):

    # Show the user's message in the chat immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user's message to display history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.spinner("Nova is thinking..."):

        # Add the new user message to Gemini history
        st.session_state.history.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

        # --- SECTION: GET NOVA'S REPLY ---
        # Try to get Nova's reply — if Gemini is busy show a friendly message
        # instead of a scary red error screen
        try:
            # Search knowledge base for relevant context
            context = get_relevant_context(prompt, chunks, embeddings)

            # If a file was uploaded, add its content to the context too
            file_context = st.session_state.file_context

            # Build enhanced prompt with knowledge base and file context injected
            enhanced_prompt = f"""
{NOVA_PERSONALITY}

## Relevant knowledge for this question:
{context}

## Uploaded file content (if any):
{file_context if file_context else "No file uploaded."}

If a file was uploaded, use its content to answer the user's question.
If relevant knowledge exists, use that too.
Otherwise just answer normally from your general knowledge.
"""

            # If an image was uploaded send it to Gemini's vision model
            # Otherwise send the normal chat history
            if st.session_state.image_bytes is not None:
                image_part = types.Part.from_bytes(
                    data=st.session_state.image_bytes,
                    mime_type=st.session_state.image_type
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=enhanced_prompt
                    ),
                    contents=[image_part, prompt]
                )
            else:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=enhanced_prompt
                    ),
                    contents=st.session_state.history
                )

            nova_reply = response.text

            # Save Nova's reply to Gemini history
            st.session_state.history.append(
                types.Content(role="model", parts=[types.Part(text=nova_reply)])
            )

        except Exception as e:
            # If anything goes wrong show a friendly message instead of crashing
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                nova_reply = "Seems like my brain is a little overloaded right now 🧠💫 Gemini's servers are busy. Give me a moment and try again!"
            elif "429" in error_msg or "quota" in error_msg.lower():
                nova_reply = "Whoa, we've been chatting a lot! 😄 I've hit my rate limit for now. Try again in a minute!"
            else:
                nova_reply = "Something unexpected happened on my end 😅 Try again in a moment!"

            # Remove the last user message from history since we didn't get a reply
            st.session_state.history.pop()

    # --- SECTION: IMAGE GENERATION ---
    # Check if Nova's reply contains an image generation request
    # If yes extract the prompt and call Pollinations.ai to generate the image
    # Pollinations.ai is free, no API key needed, works after deployment too
    
    # --- VOICE OUTPUT ---
    # If Nova speaks is on, convert her text reply to audio using gTTS
    # gTTS uses Google's servers — natural female English voice
    # We convert the audio to base64 and play it directly in the browser

    if nova_speaks and nova_reply and "[GENERATE_IMAGE:" not in nova_reply:
        try:
            # Clean the text before speaking
            # Remove emojis, symbols and fix punctuation so it sounds natural
            import re
            clean_reply = nova_reply

            # Remove all emojis and special unicode symbols
            clean_reply = re.sub(r'[^\x00-\x7F]+', ' ', clean_reply)

            # Remove markdown formatting symbols
            clean_reply = re.sub(r'\*+', '', clean_reply)   # bold/italic asterisks
            clean_reply = re.sub(r'#+\s', '', clean_reply)  # markdown headers
            clean_reply = re.sub(r'\[.*?\]\(.*?\)', '', clean_reply)  # links
            clean_reply = re.sub(r'`+', '', clean_reply)    # code backticks

            # Clean up extra spaces
            clean_reply = re.sub(r'\s+', ' ', clean_reply).strip()

            # Generate speech from cleaned text
            tts = gTTS(text=clean_reply, lang='en', tld='co.uk')  # British female voice

            # Save to memory buffer instead of a file
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Convert to base64 so browser can play it directly
            audio_b64 = base64.b64encode(audio_buffer.read()).decode()

            # Inject audio player into the page and autoplay
            st.components.v1.html(f"""
                <audio autoplay style="display:none">
                    <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
                </audio>
            """, height=0)

        except Exception:
            pass  # If voice fails silently, don't crash the app

    if "[GENERATE_IMAGE:" in nova_reply:
        image_prompt = nova_reply.split("[GENERATE_IMAGE:")[1].split("]")[0].strip()

        with st.spinner("Nova is creating your image... 🎨"):
            # Pollinations.ai works by encoding the prompt into a URL
            encoded_prompt = requests.utils.quote(image_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true"

            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                with st.chat_message("assistant"):
                    st.image(image_url, caption="Generated by Nova 🎨")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": image_url,
                    "type": "image"
                })
            else:
                with st.chat_message("assistant"):
                    st.markdown("Hmm, I had trouble generating that image 😅 Try again in a moment!")

    else:
        # Normal text reply — show in chat
        with st.chat_message("assistant"):
            st.markdown(nova_reply)

        st.session_state.messages.append({
            "role": "assistant",
            "content": nova_reply
        })