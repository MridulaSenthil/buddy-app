# -*- coding: utf-8 -*-
"""Buddy · Emotional Support (Web Deployment Version)"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import re
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification
from groq import Groq
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import uuid

st.set_page_config(
    page_title="Buddy · Emotional Support",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Lato', sans-serif; background: #B8D8F0 !important; color: #1A5276 !important; }
#MainMenu, footer, header { display: none !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 0 !important; max-width: 100% !important; }

/* ── HIDE DANGEROUS SIDEBAR BUTTON ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Collapse sidebar"] { display: none !important; }

/* ── FIX ALL INPUTS ── */
input, [data-baseweb="input"] { color: #1A5276 !important; -webkit-text-fill-color: #1A5276 !important; font-weight: 800 !important; }
input::placeholder { color: #7F8C8D !important; -webkit-text-fill-color: #7F8C8D !important; font-weight: 600 !important; }
label, .st-emotion-cache-1wivap2 { color: #1A5276 !important; font-weight: 800 !important; font-size: 1.05rem !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] { background: #D4E6F1 !important; border-right: 1.5px solid rgba(30,80,140,0.15) !important; padding: 1rem 0.8rem !important; overflow-y: hidden !important; display: flex; flex-direction: column; }
.sidebar-logo { text-align:center; padding-bottom:1rem; border-bottom:1px solid rgba(30,80,140,0.1); margin-bottom:0.8rem; }
.sidebar-logo h2 { font-family:'Nunito',sans-serif; font-size:1.8rem; color:#1A5276; margin:0; font-weight:800; }
.sidebar-logo p { font-size:0.85rem; color:#1A5276; margin:0.2rem 0 0 0; font-weight: 700; }
.sidebar-section { font-size:0.75rem; color:#1A5276; font-weight:800; letter-spacing:0.09em; text-transform:uppercase; margin:0.9rem 0 0.4rem 0; }

/* ── CHAT UI ── */
.chat-header { display:flex; align-items:center; gap:0.8rem; padding:0.9rem 1.5rem 0.8rem 1.5rem; border-bottom:1px solid rgba(30,80,140,0.12); background:#D4E6F1; border-radius:16px 16px 0 0; }
.chat-header h3 { font-family:'Nunito',sans-serif; color:#1A5276; margin:0; font-size:1.25rem; font-weight:800; }
.chat-header p { color:#1A5276; margin:0; font-size:0.85rem; font-weight: 600; }
.chat-row { display:flex; width:100%; margin-bottom:0.85rem; align-items:flex-start; gap:0.5rem; animation:fadeIn 0.25s ease; }
.chat-row.user { flex-direction:row-reverse; }
@keyframes fadeIn { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }
.avatar { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.88rem; flex-shrink:0; margin-top:3px; }
.bot-avatar { background:#1A5276; }
.user-avatar { background:#1A5276; }
.bubble { max-width:75%; padding:0.78rem 1.05rem; border-radius:18px; font-size:0.95rem; line-height:1.6; box-shadow:0 1px 5px rgba(0,0,0,0.07); word-wrap: break-word; }
.bubble-user { background:#1A5276; color:#FFFFFF !important; border-bottom-right-radius:4px; }
.bubble-bot { background:#D4E6F1; color:#1A5276 !important; border-bottom-left-radius:4px; border:1.5px solid rgba(30,80,140,0.18); font-weight: 600;}

/* ── CHAT FORMS & INPUTS ── */
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"], .stTextInput > div > div:focus-within { border: none !important; border-color: transparent !important; box-shadow: none !important; outline: none !important; background-color: transparent !important; }
.stTextInput>div>div>input { border-radius:30px !important; border:2px solid #1A5276 !important; padding:0.68rem 1.3rem !important; font-family:'Lato',sans-serif !important; font-size:0.95rem !important; background:#D4E6F1 !important; color:#1A5276 !important; caret-color: #1A5276 !important; font-weight: 800;}
.stTextInput>div>div>input:focus { outline: none !important; border-color:#1A5276 !important; box-shadow:0 0 0 3px rgba(26,82,118,0.2) !important; }
.stButton>button, .stFormSubmitButton>button { border-radius:30px !important; border:2px solid #1A5276 !important; color:#FFFFFF !important; background:#1A5276 !important; font-family:'Lato',sans-serif !important; font-size:0.9rem !important; font-weight:800 !important; padding:0.5rem 1.1rem !important; transition:all 0.2s !important; }
.stButton>button:hover, .stFormSubmitButton>button:hover { background:#0B3B60 !important; border-color:#0B3B60 !important; }

/* ── INSIGHTS DASHBOARD (BUDDY PORTAL) ── */
[data-testid="stPlotlyChart"] { background: #D4E6F1 !important; border-radius: 16px !important; padding: 0.5rem !important; box-sizing: border-box !important; box-shadow: 0 4px 10px rgba(30,80,140,0.08) !important; border: 1px solid rgba(30,80,140,0.15) !important; margin-bottom: 1rem !important; }
.insight-card { background: #D4E6F1; border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 10px rgba(30,80,140,0.08); border: 1px solid rgba(30,80,140,0.15); margin-bottom: 1.5rem; }
.module-title { font-family: 'Nunito', sans-serif; color: #1A5276; font-size: 1.25rem; font-weight: 800; margin-bottom: 0.5rem; text-align: center; }
.topic-pill { display: inline-block; background: #B8D8F0; color: #1A5276; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.95rem; font-weight: 800; margin: 0.3rem; border: 1px solid #1A5276; }
hr { border-color:rgba(30,80,140,0.15) !important; margin: 1rem 0 !important; }

/* ── PENGUIN ANIMATION ── */
@keyframes wave { 0%{transform:rotate(0deg)} 10%{transform:rotate(-15deg)} 20%{transform:rotate(10deg)} 30%{transform:rotate(-15deg)} 40%{transform:rotate(10deg)} 50%{transform:rotate(0deg)} 100%{transform:rotate(0deg)} }
@keyframes float { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-8px)} }
.penguin-container { display:inline-block; animation:float 3s ease-in-out infinite; cursor:default; margin-bottom:0.5rem; }
.penguin-wave-arm { transform-origin:50px 130px; animation:wave 2.5s ease-in-out infinite; }

/* 🔥 FIX: HIGH SPECIFICITY PADDING TO MAKE THE CARD BREATHE 🔥 */
div[data-testid="column"]:nth-of-type(2) [data-testid="stForm"] {
    background: #FFFFFF !important;
    border-radius: 24px !important;
    padding: 3.5rem 3.5rem !important;
    box-shadow: 0 12px 35px rgba(26, 82, 118, 0.12) !important;
    border: 1px solid rgba(26, 82, 118, 0.1) !important;
    margin-top: 5vh !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# CONSTANTS & CONFIG (UPDATED FOR DEPLOYMENT)
# ─────────────────────────────────────────────────
# To deploy, you must upload your emotion_bert_model_v2 folder alongside this script!
MODEL_PATH = "Mridulaaa/emotion_bert_model_v2" 

EMOTION_LABELS = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
CRISIS_KEYWORDS = ["kill","killing","suicide","isolate","hopeless","worthless","harm myself","end it","end my life","disappear","don't want to live","give up","no point","want to die","can't go on","better off dead"]
THRESHOLD = 0.35
CRISIS_THRESHOLD = 0.25

# Fetch the keys SECURELY using Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MONGO_URI    = st.secrets["MONGO_URI"]

VIBE_MAP = {
    "joy": "✨ Joyful", "optimism": "🌤️ Hopeful", "relief": "😮‍💨 Relieved", "amusement": "😂 Amused",
    "caring": "💛 Caring", "love": "❤️ Loving", "excitement": "🎉 Excited", "gratitude": "🙏 Grateful",
    "neutral": "☁️ Calm", "approval": "👍 Steady", "pride": "🦁 Proud",
    "sadness": "🌧️ Heavy", "grief": "🥀 Grieving", "disappointment": "📉 Drained", "remorse": "😔 Reflective",
    "anger": "🔥 Frustrated", "annoyance": "😤 Annoyed", "disapproval": "🙅‍♂️ Disapproving", "disgust": "🤢 Unsettled",
    "fear": "😨 Anxious", "nervousness": "😬 Nervous", "embarrassment": "😳 Self-Conscious", "confusion": "🌀 Uncertain"
}

STOPWORDS = set(["i","me","my","myself","we","our","ours","ourselves","you","your","yours","he","him","his","she","her","hers","it","its","they","them","their","theirs","what","which","who","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","im","like","really","feel","feeling","much","get","got"])

# ─────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────
for key, val in {
    "messages": [], "quick_replies": [], "pending_input": None,
    "user_name": "", "user_email": "", "name_entered": False, "input_key": 0,
    "session_id": None, "session_name": None, "viewing_session": None,
    "viewing_insights": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────
# AI MODELS & DATABASE FUNCTIONS
# ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bert():
    try:
        tok = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        mdl = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        mdl.eval()
        th_path = os.path.join(MODEL_PATH, "thresholds.npy")
        return tok, mdl, np.load(th_path) if os.path.exists(th_path) else None
    except: return None, None, None

def is_crisis(text): return any(kw in text.lower() for kw in CRISIS_KEYWORDS)

def detect_emotions(text, tok, mdl, th, top_k=2):
    crisis = is_crisis(text)
    inputs = tok(text, return_tensors="pt", max_length=128, padding=True, truncation=True)
    with torch.no_grad(): probs = torch.sigmoid(mdl(**inputs).logits).squeeze().numpy()
    scores = [(EMOTION_LABELS[i], float(probs[i])) for i in range(27)]
    # --- 🚨 ADD THESE TWO SPY LINES 🚨 ---
    top_scores = sorted(scores, key=lambda x: -x[1])[:3]
    print(f"🚨 TEXT: {text} | TOP SCORES: {top_scores}")
    # -------------------------------------
    above = sorted([(e,s) for e,s in scores if s >= (CRISIS_THRESHOLD if crisis else THRESHOLD)], key=lambda x: -x[1])
    return (above[:top_k] if above else [("neutral", float(probs[27]))]), crisis

@st.cache_resource(show_spinner=False)
def load_groq(key): return Groq(api_key=key)

def generate_response(user_msg, emotions, crisis, client, history, name):
    emotion_str = ", ".join([f"{e}({s:.0%})" for e,s in emotions])

    # Stricter prompt for the AI to ensure it formats the buttons correctly
    system = f"""You are Buddy, a warm emotional support companion talking to {name}.
    STEP 1: Write your reply to {name}. Keep it SHORT (2-3 sentences max). Acknowledge their emotion. End with ONE gentle follow-up question. NEVER start your response with "Buddy:" or "Me:".
    STEP 2: Write exactly 3 short suggested replies that {name} (THE USER) could click to answer your question.
    CRITICAL: These MUST be written in the first person ("I", "my", "me") from the USER'S perspective. YOU MUST SEPARATE THE SUGGESTIONS WITH A '|' CHARACTER.
    FORMAT EXACTLY LIKE THIS:
    [Your empathetic reply here]
    SUGGESTIONS: < YOUR OPTION 1 HERE >| <YOUR OPTION 2 HERE >| <YOUR OPTION 3 HERE>|{' CRISIS: Be very gentle, warmly suggest professional help.' if crisis else ''}"""

    history_text = "".join([f"{'User' if m['role']=='user' else 'Buddy'}: {m['content']}\n" for m in history[-6:]])
    prompt = f"{history_text}\nEmotions: {emotion_str}\nUser: \"{user_msg}\"\n\nGenerate your reply and the 3 user perspective suggestions."

    try:
        r = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"system","content":system},{"role":"user","content":prompt}], max_tokens=300)
        raw = r.choices[0].message.content.strip()
        reply, suggestions = raw, []

        if "SUGGESTIONS:" in raw:
            p = raw.split("SUGGESTIONS:", 1)
            reply = p[0].strip()
            raw_sugs = p[1].strip()

            # The Fallback Parser: Just in case the AI forgets the "|" symbol again
            if "|" in raw_sugs:
                suggestions = [s.strip() for s in raw_sugs.split("|") if s.strip() and "<opt" not in s][:3]
            else:
                # If there are no pipes, split by sentences (periods) to make neat buttons
                suggestions = [s.strip() + "." for s in raw_sugs.split(".") if len(s.strip()) > 5][:3]

        if reply.startswith("Buddy:"): reply = reply.replace("Buddy:", "", 1).strip()
        elif reply.startswith("**Buddy**:"): reply = reply.replace("**Buddy**:", "", 1).strip()
        return reply, suggestions
    except: return "I'm here with you. What would you like to share?", []

@st.cache_resource(show_spinner=False)
def get_mongo(uri):
    try:
        c = MongoClient(uri, serverSelectionTimeoutMS=3000)
        c.server_info()
        return c["mindease_db"]["chat_sessions"]
    except: return None

def save_msg(col, name, email, session_id, session_name, user_msg, bot_reply, emotions, crisis):
    if col is None: return
    try:
        col.insert_one({"user_name": name, "user_email": email, "session_id": session_id, "session_name": session_name, "timestamp": datetime.utcnow(), "user_message": user_msg, "bot_reply": bot_reply, "emotions": [{"label": e, "score": round(s,4)} for e,s in emotions], "is_crisis": crisis})
    except: pass

def get_all_sessions(col, email):
    if col is None: return []
    try:
        pipeline = [{"$match": {"user_email": email}}, {"$sort": {"timestamp": -1}}, {"$group": {"_id": "$session_id", "session_name": {"$first": "$session_name"}, "latest": {"$first": "$timestamp"}}}, {"$sort": {"latest": -1}}]
        return list(col.aggregate(pipeline))
    except: return []

def load_session_messages(col, email, session_id):
    if col is None: return []
    try:
        docs = list(col.find({"user_email": email, "session_id": session_id}).sort("timestamp", 1))
        messages = []
        for doc in docs:
            messages.append({"role": "user", "content": doc["user_message"]})
            messages.append({"role": "bot", "content": doc["bot_reply"], "emotions": [(e["label"], e["score"]) for e in doc.get("emotions", [])], "is_crisis": doc.get("is_crisis", False)})
        return messages
    except: return []

def extract_topics(text_list):
    words = []
    for text in text_list:
        clean_text = re.sub(r'[^\w\s]', '', str(text).lower())
        words.extend([w for w in clean_text.split() if w not in STOPWORDS and len(w) > 2])
    counter = Counter(words)
    return [word for word, _ in counter.most_common(5)]

# SVGS
PENGUIN_SVG_WAVEEZY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="160" height="160"><path d="M 140 130 C 180 140, 185 180, 160 200" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/><g class="penguin-wave-arm"><path d="M 60 130 C 10 110, 5 50, 30 45 C 50 40, 65 80, 70 115" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/></g><path d="M 40 200 C 30 100, 45 45, 100 45 C 155 45, 170 100, 160 200 Z" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/><path d="M 100 75 C 70 55, 45 75, 55 120 C 60 170, 75 195, 100 195 C 125 195, 140 170, 145 120 C 155 75, 130 55, 100 75 Z" fill="#FFFFFF" stroke="#2A3B5C" stroke-width="6" stroke-linejoin="round"/><path d="M 90 45 C 75 20, 105 25, 100 38 C 115 25, 130 30, 110 45" fill="none" stroke="#2A3B5C" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/><ellipse cx="78" cy="100" rx="11" ry="15" fill="#2A3B5C"/><ellipse cx="122" cy="100" rx="11" ry="15" fill="#2A3B5C"/><circle cx="75" cy="94" r="4" fill="#FFFFFF"/><circle cx="119" cy="94" r="4" fill="#FFFFFF"/><circle cx="82" cy="106" r="2" fill="#FFFFFF"/><circle cx="126" cy="106" r="2" fill="#FFFFFF"/><path d="M 88 112 C 95 106, 105 106, 112 112 C 115 120, 105 128, 100 128 C 95 128, 85 120, 88 112 Z" fill="#F5A623" stroke="#2A3B5C" stroke-width="4" stroke-linejoin="round"/><path d="M 91 116 C 95 118, 105 118, 109 116" fill="none" stroke="#2A3B5C" stroke-width="3" stroke-linecap="round"/></svg>"""
PENGUIN_SVG_SMALL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="45" height="45"><path d="M 140 130 C 180 140, 185 180, 160 200" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/><path d="M 60 130 C 10 110, 5 50, 30 45 C 50 40, 65 80, 70 115" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/><path d="M 40 200 C 30 100, 45 45, 100 45 C 155 45, 170 100, 160 200 Z" fill="#8A95A5" stroke="#2A3B5C" stroke-width="6"/><path d="M 100 75 C 70 55, 45 75, 55 120 C 60 170, 75 195, 100 195 C 125 195, 140 170, 145 120 C 155 75, 130 55, 100 75 Z" fill="#FFFFFF" stroke="#2A3B5C" stroke-width="6" stroke-linejoin="round"/><path d="M 90 45 C 75 20, 105 25, 100 38 C 115 25, 130 30, 110 45" fill="none" stroke="#2A3B5C" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/><ellipse cx="78" cy="100" rx="11" ry="15" fill="#2A3B5C"/><ellipse cx="122" cy="100" rx="11" ry="15" fill="#2A3B5C"/><circle cx="75" cy="94" r="4" fill="#FFFFFF"/><circle cx="119" cy="94" r="4" fill="#FFFFFF"/><circle cx="82" cy="106" r="2" fill="#FFFFFF"/><circle cx="126" cy="106" r="2" fill="#FFFFFF"/><path d="M 88 112 C 95 106, 105 106, 112 112 C 115 120, 105 128, 100 128 C 95 128, 85 120, 88 112 Z" fill="#F5A623" stroke="#2A3B5C" stroke-width="4" stroke-linejoin="round"/><path d="M 91 116 C 95 118, 105 118, 109 116" fill="none" stroke="#2A3B5C" stroke-width="3" stroke-linecap="round"/></svg>"""

tok, bert_mdl, th = load_bert()
bert_ready = bert_mdl is not None
groq_client = load_groq(GROQ_API_KEY)
mongo_col = get_mongo(MONGO_URI)

# ─────────────────────────────────────────────────
# SCREEN 1: THE PERFECT LOGIN CARD
# ─────────────────────────────────────────────────
if not st.session_state.name_entered:
    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        with st.form("login_form"):
            st.markdown(f'''
            <div style="text-align: center;">
                <div class="penguin-container" style="margin-bottom: 0;">{PENGUIN_SVG_WAVEEZY}</div>
                <h1 style="font-family:'Nunito',sans-serif; font-size:2.8rem; color:#1A5276; margin-bottom:0.2rem; font-weight:800; line-height: 1.1;">Buddy</h1>
                <p style="color:#7F8C8D; font-size:1.05rem; margin-bottom:2rem; font-weight:600;">Sign in to your personal wellness space.</p>
            </div>
            ''', unsafe_allow_html=True)

            name = st.text_input("Full Name", placeholder="Name")
            email = st.text_input("Email ID", placeholder="Email ID")

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Start Chatting →", use_container_width=True)

            if submit:
                if name.strip() and email.strip():
                    st.session_state.user_name = name.strip()
                    st.session_state.user_email = email.strip().lower()
                    st.session_state.name_entered = True
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("Please enter both your Name and Email ID to continue.")
    st.stop()

# ─────────────────────────────────────────────────
# SCREEN 2: BUDDY INSIGHTS PORTAL
# ─────────────────────────────────────────────────
if st.session_state.viewing_insights:
    col_head, col_vitals, col_btn = st.columns([2, 3, 1])
    with col_head:
        st.markdown(f"<h2 style='color:#1A5276; font-family:Nunito, sans-serif; margin-bottom: 0;'>Hi, {st.session_state.user_name}! 👋</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#1A5276; font-size: 1.1rem; font-weight: 800; margin-top: 0;'>Your Wellness Pulse & Reflections.</p>", unsafe_allow_html=True)

    docs = list(mongo_col.find({"user_email": st.session_state.user_email}))
    if not docs:
        st.info("Buddy needs a little more time with you! Start chatting to unlock your insights.")
        if st.button("← Back to Chat"):
            st.session_state.viewing_insights = False
            st.rerun()
        st.stop()

    df = pd.DataFrame(docs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    ist = pytz.timezone('Asia/Kolkata')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ist)
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    today = datetime.now(ist).date()

    total_messages = len(df)
    unique_dates = sorted(df['date'].unique(), reverse=True)
    streak = 0
    if unique_dates:
        start_date = unique_dates[0]
        if (today - start_date).days <= 1:
            curr = start_date
            for d in unique_dates:
                if d == curr:
                    streak += 1
                    curr -= timedelta(days=1)
                else: break

    emo_rows = []
    for _, row in df.iterrows():
        emotions_data = row.get('emotions', [])
        if isinstance(emotions_data, list):
            for e in emotions_data:
                label, score = 'neutral', 0.0
                if isinstance(e, dict):
                    label = e.get('label', 'neutral')
                    score = e.get('score', 0.0)
                elif isinstance(e, list) and len(e) >= 2:
                    label = str(e[0])
                    score = float(e[1])
                emo_rows.append({
                    'timestamp': row['timestamp'], 'date': row['date'], 'hour': row['hour'],
                    'message': row.get('user_message', ''), 'label': label, 'score': score
                })
    edf = pd.DataFrame(emo_rows)

    with col_vitals:
        vc1, vc2 = st.columns(2)
        vc1.markdown(f"<div class='insight-card' style='padding: 1rem;'><h4 style='color:#1A5276; font-size:1.1rem; margin:0; text-transform:uppercase; font-weight:800;'>Check-in Streak</h4><div style='font-size:2.2rem; font-weight:800; color:#1A5276;'>🔥 {streak} Days</div></div>", unsafe_allow_html=True)
        vc2.markdown(f"<div class='insight-card' style='padding: 1rem;'><h4 style='color:#1A5276; font-size:1.1rem; margin:0; text-transform:uppercase; font-weight:800;'>Messages Shared</h4><div style='font-size:2.2rem; font-weight:800; color:#1A5276;'>💬 {total_messages}</div></div>", unsafe_allow_html=True)

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Continue Chatting", use_container_width=True):
            st.session_state.viewing_insights = False
            st.rerun()

    st.markdown("<hr style='margin-top: 0 !important;'>", unsafe_allow_html=True)

    col_hub, col_trend, col_deep = st.columns([1, 1.4, 1])

    with col_hub:
        st.markdown("<div class='module-title'>📅 Reflection Calendar</div>", unsafe_allow_html=True)
        selected_date = st.date_input("Select a day to review:", today, label_visibility="collapsed")

        st.markdown("<br><div class='module-title'>🎯 Daily Snapshot</div>", unsafe_allow_html=True)
        daily_edf = edf[edf['date'] == selected_date]
        if not daily_edf.empty:
            pie_data = daily_edf.groupby('label')['score'].sum().reset_index()
            fig_pie = px.pie(pie_data, values='score', names='label', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, height=280)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', insidetextfont=dict(color='#1A5276', weight='bold'))
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No chats recorded on this day.")

    with col_trend:
        st.markdown("<div class='module-title'>📊 Weekly Emotion Trends</div>", unsafe_allow_html=True)
        week_ago = selected_date - timedelta(days=7)
        recent_edf = edf[(edf['date'] > week_ago) & (edf['date'] <= selected_date)]

        if not recent_edf.empty:
            top_emotions = recent_edf.groupby('label')['score'].sum().nlargest(4).index
            bar_data = recent_edf[recent_edf['label'].isin(top_emotions)].groupby(['date', 'label'])['score'].mean().reset_index()
            fig_bar = px.bar(bar_data, x='date', y='score', color='label', barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(
                margin=dict(t=40, b=20, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Intensity",
                legend_title="",
                height=350,
                font=dict(color='#1A5276'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#1A5276', weight='bold')
                )
            )
            fig_bar.update_xaxes(tickformat="%b %d", dtick="86400000", title="", tickfont=dict(color='#1A5276', weight='bold'))
            fig_bar.update_yaxes(tickfont=dict(color='#1A5276', weight='bold'), title_font=dict(color='#1A5276', weight='bold'))
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Not enough data for the week leading up to this date.")

    with col_deep:
        if not daily_edf.empty:
            top_emotion = daily_edf.groupby('label')['score'].sum().idxmax()
            vibe_text = VIBE_MAP.get(top_emotion, f"🌤️ {top_emotion.capitalize()}")
        else:
            vibe_text = "☁️ No data"

        vibe_html = f"""
        <div class="insight-card">
            <h4 style="color:#1A5276; font-size:1.1rem; margin:0; text-transform:uppercase; letter-spacing:1px; font-weight:800;">Dominant Vibe</h4>
            <div style="font-size:2.8rem; font-weight:800; color:#1A5276; margin: 0.8rem 0;">{vibe_text}</div>
            <p style="font-size: 0.95rem; color:#1A5276; margin:0; font-weight:700;">Your dominant feeling on {selected_date.strftime('%b %d')}</p>
        </div>
        """
        st.markdown(vibe_html, unsafe_allow_html=True)

        pill_html = ""
        if not recent_edf.empty:
            strong_emotions = ['anger', 'sadness', 'fear', 'nervousness', 'annoyance', 'stress', 'disappointment']
            target_msgs = recent_edf[recent_edf['label'].isin(strong_emotions)]['message'].unique()
            if len(target_msgs) > 0:
                top_words = extract_topics(target_msgs)
                if top_words: pill_html = "".join([f"<span class='topic-pill'># {word}</span>" for word in top_words])
                else: pill_html = "<p style='color:#1A5276; font-weight:700;'>Not enough vocabulary detected yet.</p>"
            else: pill_html = "<p style='color:#1A5276; font-weight:700;'>No high-stress topics detected recently! Great job.</p>"
        else: pill_html = "<p style='color:#1A5276; font-weight:700;'>Select a date with chat history to see topics.</p>"

        topics_html = f"""
        <div class="insight-card">
            <h4 style="color:#1A5276; font-size:1.1rem; margin:0; text-transform:uppercase; letter-spacing:1px; font-weight:800;">Common Topics</h4>
            <p style="color:#1A5276; font-size: 0.9rem; margin-top:0.3rem; margin-bottom: 1.2rem; font-weight: 600;">Words used during high-intensity moments.</p>
            <div>{pill_html}</div>
        </div>
        """
        st.markdown(topics_html, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────
# SCREEN 3: NORMAL CHAT & SIDEBAR
# ─────────────────────────────────────────────────
all_sessions = get_all_sessions(mongo_col, st.session_state.user_email)

with st.sidebar:
    st.markdown(f'<div class="sidebar-logo"><div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;">{PENGUIN_SVG_SMALL}</div><h2>Buddy</h2><p>Hi {st.session_state.user_name}! 👋</p></div>', unsafe_allow_html=True)

    if st.button("📊 Buddy Portal", use_container_width=True):
        st.session_state.viewing_insights = True
        st.rerun()

    st.markdown("<hr style='margin: 0.5rem 0 !important;'>", unsafe_allow_html=True)

    if st.button("✦  New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_name = None
        st.session_state.messages = []
        st.session_state.quick_replies = []
        st.session_state.input_key += 1
        st.rerun()

    if all_sessions:
        st.markdown('<p class="sidebar-section">Conversations</p>', unsafe_allow_html=True)
        for sess in all_sessions:
            sid = sess["_id"]
            sname = sess["session_name"] or "New conversation"
            is_active = (sid == st.session_state.session_id)
            if st.button(f"{'▶ ' if is_active else '💬 '}{sname}", key=f"sess_{sid}", use_container_width=True):
                if not is_active:
                    st.session_state.messages = load_session_messages(mongo_col, st.session_state.user_email, sid)
                    st.session_state.session_id = sid
                    st.session_state.session_name = sname
                    st.session_state.quick_replies = []
                    st.session_state.input_key += 1
                    st.rerun()

    st.markdown("<div style='flex-grow: 1;'></div><hr style='margin: 1rem 0 0.5rem 0 !important;'>", unsafe_allow_html=True)
    if st.button("← Switch User", use_container_width=True):
        for k in ["messages","quick_replies","user_name","user_email","name_entered","input_key","session_id","session_name","viewing_insights"]: st.session_state.pop(k, None)
        st.rerun()

# --- MAIN CHAT UI ---
st.markdown(f'<div class="chat-header"><div style="width:50px;height:50px;display:flex;align-items:center;justify-content:center;">{PENGUIN_SVG_SMALL}</div><div><h3>Buddy</h3><p>Always here to listen, {st.session_state.user_name} 🐧</p></div></div>', unsafe_allow_html=True)
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

st.markdown(f'<div class="chat-row"><div class="avatar bot-avatar">🐧</div><div class="bubble bubble-bot">Hi {st.session_state.user_name}! 👋 I\'m Buddy — here to listen without any judgment. How are you feeling today? 🐧</div></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-row user"><div class="avatar user-avatar">🙂</div><div class="bubble bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
    elif msg["role"] == "bot":
        st.markdown(f'<div class="chat-row"><div class="avatar bot-avatar">🐧</div><div class="bubble bubble-bot">{msg["content"]}</div></div>', unsafe_allow_html=True)
        if msg.get("is_crisis", False):
            st.markdown('<div class="crisis-card"><h4>💛 You\'re not alone in this</h4>Please reach out to someone right now.<br><br><strong>iCall (India):</strong> 9152987821<br><strong>Vandrevala Foundation:</strong> 1860-2662-345 (24/7)</div><div class="breathing-box" style="margin-top:0.35rem;">🌬️ <strong>Breathe with me</strong> — In for 4... hold for 4... out for 6.<br><em>You are safe right now.</em></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.quick_replies:
    st.markdown('<p class="suggestions-label">Suggested replies</p>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.quick_replies))
    for i, s in enumerate(st.session_state.quick_replies):
        with cols[i]:
            if st.button(s, key=f"qr_{i}_{s[:8]}"):
                st.session_state.pending_input = s
                st.session_state.quick_replies = []
                st.session_state.input_key += 1
                st.rerun()

st.markdown("---")

user_input, send_clicked = "", False
if st.session_state.pending_input:
    user_input, st.session_state.pending_input, send_clicked = st.session_state.pending_input, None, True
else:
    with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
        c1, c2 = st.columns([6, 1])
        with c1: user_input = st.text_input("Message", placeholder=f"Share how you're feeling, {st.session_state.user_name}...", label_visibility="collapsed", autocomplete="off")
        with c2: send_clicked = st.form_submit_button("Send →", use_container_width=True)

if send_clicked and user_input and user_input.strip():
    text = user_input.strip()
    if not st.session_state.session_name: st.session_state.session_name = text[:40] + ("..." if len(text) > 40 else "")
    st.session_state.messages.append({"role": "user", "content": text})

    emotions, crisis_flag = detect_emotions(text, tok, bert_mdl, th) if bert_ready else ([("neutral", 0.5)], is_crisis(text))

    with st.spinner(""):
        bot_reply, suggestions = generate_response(text, emotions, crisis_flag, groq_client, st.session_state.messages, st.session_state.user_name)

    st.session_state.messages.append({"role": "bot", "content": bot_reply, "emotions": emotions, "is_crisis": crisis_flag})

    save_msg(mongo_col, st.session_state.user_name, st.session_state.user_email, st.session_state.session_id, st.session_state.session_name, text, bot_reply, emotions, crisis_flag)

    st.session_state.quick_replies = suggestions
    st.session_state.input_key += 1
    st.rerun()
