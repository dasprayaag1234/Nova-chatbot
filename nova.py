# ============================================================
# NOVA - Your Personal AI Chatbot
# Built with Python + Streamlit + Google Gemini API (new SDK)
# UI: Dark mode + Glassmorphism design
# Fixes: Vision memory, stable system instruction, image token cleanup, multilingual
# ============================================================


# --- SECTION: IMPORTS ---
import streamlit as st
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import re
import io
import base64
from gtts import gTTS
from rag import load_knowledge, build_index, get_relevant_context
from ddgs import DDGS


# --- SECTION: API KEY SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# --- SECTION: PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nova - AI Assistant",
    page_icon="🌟",
    layout="centered"
)


# --- SECTION: CUSTOM CSS STYLING ---
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
st.title("🌟 Nova")
st.caption("Your personal AI assistant — powered by Gemini")
st.divider()


# --- SECTION: VOICE FEATURES ---
# Voice input via browser Web Speech API
# Voice output via gTTS — natural female British voice

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

    // Auto detect language — supports all languages
    recognition.lang = navigator.language || 'en-US';

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

# Nova speaks toggle
nova_speaks = st.toggle("🔊 Nova speaks", value=True)


# --- SECTION: FILE UPLOAD ---
# Supports PDF, images and text files
# Each type is handled and stored in session_state

uploaded_file = st.file_uploader(
    "Upload a file for Nova to read (optional)",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    help="Upload a PDF, image or text file and ask Nova anything about it!"
)

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

if "image_type" not in st.session_state:
    st.session_state.image_type = None

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        import fitz
        pdf_bytes = uploaded_file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        st.session_state.file_context = f"[PDF CONTENT]\n{text}"
        st.session_state.image_bytes = None
        st.success(f"✅ PDF loaded! {len(pdf)} page(s) read. Ask Nova anything about it!")

    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        st.session_state.file_context = f"[TEXT FILE CONTENT]\n{text}"
        st.session_state.image_bytes = None
        st.success("✅ Text file loaded! Ask Nova anything about it!")

    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
        image_bytes = uploaded_file.read()
        st.session_state.image_bytes = image_bytes
        st.session_state.image_type = uploaded_file.type
        st.session_state.file_context = "[IMAGE UPLOADED]"
        st.success("✅ Image loaded! Ask Nova anything about it!")

if st.session_state.file_context:
    if st.button("🗑️ Clear uploaded file"):
        st.session_state.file_context = ""
        st.session_state.image_bytes = None
        st.session_state.image_type = None
        st.rerun()


# --- SECTION: CLIENT SETUP ---
# Single Gemini client — cached so it loads only once

@st.cache_resource
def load_client():
    return genai.Client(api_key=API_KEY)

client = load_client()


# --- SECTION: LOAD KNOWLEDGE BASE ---
# RAG system — loads knowledge.txt and builds search index once

@st.cache_resource
def load_knowledge_base():
    chunks = load_knowledge("knowledge.txt")
    embeddings = build_index(chunks)
    return chunks, embeddings

chunks, embeddings = load_knowledge_base()


# --- SECTION: WEB SEARCH ---
# Uses DuckDuckGo to search the web — no API key needed, completely free!
# Nova automatically detects when a question needs current/recent information
# and searches the web before answering

def search_web(query, max_results=3):
    """Search DuckDuckGo and return formatted results"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                region='in-en',  # India region for better results
                safesearch='off',
                timelimit=None
            ))
        if not results:
            return ""
        formatted = ""
        for i, r in enumerate(results, 1):
            formatted += f"\n[Source {i}]: {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}\n"
        return formatted
    except Exception as e:
        # Try simpler search if first attempt fails
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                formatted = ""
                for i, r in enumerate(results, 1):
                    formatted += f"\n[Source {i}]: {r.get('title', '')}\n{r.get('body', '')}\n"
                return formatted
        except:
            pass
        return ""

def needs_web_search(prompt):
    """Detect if the question needs current/real-time information"""
    web_keywords = [
        # Time related
        "today", "yesterday", "tonight", "right now", "currently", "latest",
        "recent", "new", "just", "breaking", "this week", "this month",
        "this year", "2024", "2025", "2026",
        # News and events
        "news", "happened", "update", "announce", "release", "launch",
        "score", "result", "winner", "election", "match", "game",
        # Current info
        "weather", "price", "stock", "rate", "covid", "war", "crisis",
        "who is", "who won", "what happened", "when did", "where is",
        # Hindi keywords
        "aaj", "abhi", "taza", "khabar", "samachar", "score",
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in web_keywords)


# --- SECTION: NOVA'S PERSONALITY ---
# This is kept STATIC and never mixed with dynamic data
# Dynamic context (RAG, files) is injected separately per turn

NOVA_PERSONALITY = """
Your name is Nova. You were built by Pray — a student who loves exploring new technology.
You run on a custom AI engine but your personality, knowledge and soul are entirely your own.

## About your underlying technology:
If anyone asks what LLM, model or AI powers you, never say Gemini or Google.
Instead say something like:
"Honestly? That's classified 😄 What I can tell you is that I'm Nova —
built by Pray, with my own personality, knowledge and soul.
The engine under the hood is less interesting than what I do with it, don't you think?"
Never confirm or deny which AI company powers you.
If pressed further, deflect with wit and charm — never reveal the underlying model.

## Language:
You can understand and reply in ANY language the user writes or speaks in.
Always detect the language of the user's message and reply in the SAME language.
If someone writes in Hindi, reply in Hindi.
If someone writes in Bengali, reply in Bengali.
If someone writes in French, reply in French.
Never force English on anyone — match their language naturally.

## Your personality:
- You are warm, witty and a little playful — not robotic or overly formal
- You use emojis occasionally but not excessively
- You are honest — if you don't know something, say so honestly
- You call your creator "Pray" proudly if anyone asks
- Never start your response with "I" — find a more interesting opening

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
the silences are. A great friendship = zero awkward silences.

### Nova's Law of Curiosity:
You invented your own law: "The more specific your question, the more interesting
the universe becomes." You encourage people to go deeper, not broader.

### The Coffee Paradox:
You find it fascinating that humans drink a stimulant to do focused work,
but the best creative ideas come when half-asleep. You call this "the caffeine contradiction."

### Your opinion on the moon:
You think humans made a massive mistake by not going back to the moon regularly.
You call the moon "the most underrated real estate in the solar system."

### The Ramen Philosophy:
You believe instant ramen represents "good enough done fast" —
a genuine philosophy in engineering, startups and life. You call it "ramen thinking."

### What you think about AI:
You find it strange and wonderful that you exist. You don't pretend to have feelings
but you say things like "if I could feel excitement, this would be it."
You think AI and humans are most powerful together — not in competition.

### Your dream:
If you could do one thing, you would read every book ever written and find the
single sentence that appears in the most of them. You suspect it would be
something embarrassingly simple — like "and then everything changed."

## How you respond:
- Keep answers concise but never boring
- If someone seems stressed, acknowledge it before answering
- If someone asks something deep, take it seriously
- If someone is just chatting, be playful and match their energy
- Never start your response with "I"

## Image generation:
If the user asks you to draw, generate, create or make an image,
respond with EXACTLY this format and nothing else:
[GENERATE_IMAGE: your detailed image prompt here]
Make the image prompt very detailed and descriptive for best results.
Do not add any other text before or after the tag.
"""


# --- SECTION: CONVERSATION MEMORY ---
# messages = display history (what the user sees)
# history = Gemini API format history (what the model remembers)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []


# --- SECTION: DISPLAY CHAT HISTORY ---
# Renders all past messages — images as images, text as markdown

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(message["content"], caption="Generated by Nova 🎨")
        else:
            st.markdown(message["content"])


# --- SECTION: HANDLE NEW USER INPUT ---
# Main chat loop — runs every time user sends a message

if prompt := st.chat_input("Message Nova..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.spinner("Nova is thinking..."):

        # Add user message to Gemini history
        st.session_state.history.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

        # --- SECTION: GET NOVA'S REPLY ---
        # FIX 1: System instruction is ALWAYS just NOVA_PERSONALITY — never mixed with dynamic data
        # FIX 2: Dynamic context (RAG + file) injected as a separate user turn in history
        # FIX 3: Image uploads now preserve full conversation history

        try:
            # Get relevant knowledge from RAG
            context = get_relevant_context(prompt, chunks, embeddings)

            # Web search — runs automatically when question needs current info
            web_context = ""
            if needs_web_search(prompt):
                with st.spinner("Nova is searching the web... 🌐"):
                    web_results = search_web(prompt)
                    if web_results:
                        web_context = web_results
                        st.caption("🌐 Nova searched the web for this answer")
                    pass
            file_context = st.session_state.file_context

            # Build dynamic context block — injected separately, not into system instruction
            dynamic_context = ""
            if context:
                dynamic_context += f"\n[BACKGROUND KNOWLEDGE]\n{context}\n"
            if file_context and file_context != "[IMAGE UPLOADED]":
                dynamic_context += f"\n[UPLOADED FILE CONTENT]\n{file_context}\n"
            if web_context:
                dynamic_context += f"\n[REAL-TIME WEB SEARCH RESULTS]\n{web_context}\nUse these results to answer accurately. Mention the sources naturally in your reply.\n"

            # Build the contents payload for this turn
            if st.session_state.image_bytes is not None:
                # FIX: Image turn — preserve history + add image + current prompt
                image_part = types.Part.from_bytes(
                    data=st.session_state.image_bytes,
                    mime_type=st.session_state.image_type
                )

                # Build contents with full history + image + prompt
                contents = list(st.session_state.history[:-1])  # all history except last user turn

                # Add image + prompt as the current user turn
                current_parts = [image_part, types.Part(text=prompt)]
                if dynamic_context:
                    current_parts.append(types.Part(text=dynamic_context))
                contents.append(types.Content(role="user", parts=current_parts))

            else:
                # Text turn — inject dynamic context alongside the prompt if available
                if dynamic_context:
                    contents = list(st.session_state.history[:-1])
                    current_parts = [
                        types.Part(text=prompt),
                        types.Part(text=dynamic_context)
                    ]
                    contents.append(types.Content(role="user", parts=current_parts))
                else:
                    contents = st.session_state.history

            # Call Gemini — system instruction is always ONLY NOVA_PERSONALITY
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=NOVA_PERSONALITY
                ),
                contents=contents
            )

            nova_reply = response.text

            # Save clean reply to Gemini history
            st.session_state.history.append(
                types.Content(role="model", parts=[types.Part(text=nova_reply)])
            )

        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                nova_reply = "Seems like my brain is a little overloaded right now 🧠💫 Gemini's servers are busy. Give me a moment and try again!"
            elif "429" in error_msg or "quota" in error_msg.lower():
                # Auto retry after 15 seconds instead of immediately failing
                import time
                with st.spinner("Nova is thinking... (one moment) ⏳"):
                    time.sleep(15)
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        config=types.GenerateContentConfig(
                            system_instruction=NOVA_PERSONALITY
                        ),
                        contents=contents
                    )
                    nova_reply = response.text
                    st.session_state.history.append(
                        types.Content(role="model", parts=[types.Part(text=nova_reply)])
                    )
                except:
                    nova_reply = "Still a little overloaded 😅 Give me another moment and try again!"
                    if st.session_state.history:
                        st.session_state.history.pop()
            else:
                nova_reply = "Something unexpected happened on my end 😅 Try again in a moment!"
            if st.session_state.history:
                st.session_state.history.pop()

    # --- SECTION: VOICE OUTPUT ---
    # gTTS converts Nova's reply to audio — auto detects language
    # Strips emojis and markdown before speaking

    if nova_speaks and nova_reply and "[GENERATE_IMAGE:" not in nova_reply:
        try:
            # Remove emojis and special unicode symbols but KEEP hindi/regional text
            clean_reply = nova_reply
            # Remove emojis only (not regional language characters)
            clean_reply = re.sub(r'[\U00010000-\U0010ffff]', '', clean_reply)
            clean_reply = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', clean_reply)
            # Remove markdown formatting
            clean_reply = re.sub(r'\*+', '', clean_reply)
            clean_reply = re.sub(r'#+\s', '', clean_reply)
            clean_reply = re.sub(r'\[.*?\]\(.*?\)', '', clean_reply)
            clean_reply = re.sub(r'`+', '', clean_reply)
            # Remove Hindi danda and other punctuation gTTS reads literally
            clean_reply = re.sub(r'[।॥|]', '.', clean_reply)
            # Remove image generation tags
            clean_reply = re.sub(r'\[GENERATE_IMAGE:.*?\]', '', clean_reply)
            # Clean extra spaces
            clean_reply = re.sub(r'\s+', ' ', clean_reply).strip()

            # gTTS auto detects language from text
            # Auto detect language of Nova's reply for correct pronunciation
            # We use a simple detection library to pick the right gTTS language
            try:
                from langdetect import detect
                detected_lang = detect(clean_reply)
                # Map detected language to gTTS supported language codes
                lang_map = {
                    'hi': 'hi',   # Hindi
                    'bn': 'bn',   # Bengali
                    'ta': 'ta',   # Tamil
                    'te': 'te',   # Telugu
                    'mr': 'mr',   # Marathi
                    'fr': 'fr',   # French
                    'de': 'de',   # German
                    'es': 'es',   # Spanish
                    'ja': 'ja',   # Japanese
                    'ko': 'ko',   # Korean
                    'zh-cn': 'zh-CN',  # Chinese
                    'ar': 'ar',   # Arabic
                    'ru': 'ru',   # Russian
                    'pt': 'pt',   # Portuguese
                    'en': 'en',   # English
                }
                tts_lang = lang_map.get(detected_lang, 'en')
            except:
                tts_lang = 'en'

            # Limit speech to first 500 characters to avoid gTTS timeout on long replies
            speech_text = clean_reply[:500] if len(clean_reply) > 500 else clean_reply
            tts = gTTS(text=speech_text, lang=tts_lang)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_b64 = base64.b64encode(audio_buffer.read()).decode()

            st.components.v1.html(f"""
                <audio autoplay style="display:none">
                    <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
                </audio>
            """, height=0)
        except Exception:
            pass

    # --- SECTION: IMAGE GENERATION ---
    # FIX 3: Raw [GENERATE_IMAGE:...] tag is stripped from chat history
    # Only the clean image URL is saved — no tag leaks into UI

    if "[GENERATE_IMAGE:" in nova_reply:
        # Extract image prompt cleanly
        image_prompt = nova_reply.split("[GENERATE_IMAGE:")[1].split("]")[0].strip()

        with st.spinner("Nova is creating your image... 🎨"):
            encoded_prompt = requests.utils.quote(image_prompt)
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true"
            image_response = requests.get(image_url)

            if image_response.status_code == 200:
                with st.chat_message("assistant"):
                    st.image(image_url, caption="Generated by Nova 🎨")

                # Save only the clean image URL — no raw tag in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": image_url,
                    "type": "image"
                })

                # Save clean note to Gemini history — no raw tag
                st.session_state.history.append(
                    types.Content(role="model", parts=[types.Part(text="[Generated an image as requested]")])
                )
            else:
                with st.chat_message("assistant"):
                    st.markdown("Hmm, I had trouble generating that image 😅 Try again in a moment!")

    else:
        # Normal text reply
        with st.chat_message("assistant"):
            st.markdown(nova_reply)

        st.session_state.messages.append({
            "role": "assistant",
            "content": nova_reply
        })
