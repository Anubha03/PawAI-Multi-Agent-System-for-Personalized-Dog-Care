"""
🐾 PawAI — Streamlit Chat UI 

"""

import requests
import streamlit as st
import time

import os
API_BASE = os.environ.get("FLASK_PUBLIC_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🐾 PawAI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --primary-bg: #faf7f2;
    --primary-white: #fff;
    --primary-accent: #d4a574;
    --primary-accent-dark: #b8895a;
    --primary-dark: #2d1a0e;
    --border-light: #ede8e0;
    --border-lighter: #e8ddd0;
    --text-light: #a3a3a3;
    --text-muted: #6b5d52;
}

* {
    font-family: 'DM Sans', sans-serif;
}

html, body, [class*="css"], [data-testid*="StyledDataFrame"] {
    background-color: var(--primary-bg) !important;
}

/* Main container */
.stApp {
    background-color: var(--primary-bg);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--primary-white) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    color: var(--primary-dark);
    margin: 0;
    padding: 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    color: var(--primary-dark);
    margin-top: 10px;
}

/* Input fields in sidebar */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background-color: var(--primary-bg) !important;
    border: 1px solid var(--border-lighter) !important;
    border-radius: 8px !important;
    color: var(--primary-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] select:focus {
    border-color: var(--primary-accent) !important;
    box-shadow: 0 0 0 3px rgba(212,165,116,0.15) !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--primary-accent) !important;
    color: var(--primary-white) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: var(--primary-accent-dark) !important;
    transform: scale(1.02);
}

.stButton > button:disabled {
    background-color: var(--border-lighter) !important;
    color: var(--text-light) !important;
}

/* Checkbox & Radio */
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    font-size: 13px;
    color: var(--text-muted);
}

/* Suggestion buttons - pill shaped */
.suggestion-btn {
    background-color: var(--primary-white) !important;
    border: 1px solid var(--border-lighter) !important;
    border-radius: 20px !important;
    padding: 8px 14px !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
}

.suggestion-btn:hover {
    background-color: #f5ede0 !important;
    border-color: var(--primary-accent) !important;
}

/* Spinner & Loading */
.stSpinner {
    color: var(--primary-accent);
}

/* Divider */
hr {
    border-color: var(--border-light) !important;
}

/* Main content area */
[data-testid="stExpander"] {
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
}

/* Text styling */
h1, h2 {
    font-family: 'Playfair Display', serif;
    color: var(--primary-dark);
}

p {
    color: var(--primary-dark);
}

.caption {
    color: var(--text-light);
    font-size: 11px;
}

/* Success/Error boxes */
.stNotification, [data-testid="stNotification"] {
    border-radius: 8px;
}

/* Message bubble styling */
.msg-user {
    background-color: var(--primary-accent);
    color: var(--primary-white);
    padding: 12px 16px;
    border-radius: 18px 4px 18px 18px;
    margin-bottom: 12px;
    animation: fadeUp 0.3s ease;
}

.msg-assistant {
    background-color: var(--primary-white);
    color: var(--primary-dark);
    padding: 12px 16px;
    border: 1px solid var(--border-light);
    border-radius: 4px 18px 18px 18px;
    margin-bottom: 12px;
    animation: fadeUp 0.3s ease;
}

.msg-badges {
    margin-bottom: 8px;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.msg-content {
    line-height: 1.6;
    font-size: 14px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

/* Agent badges */
.badge-health { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
.badge-nutrition { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.badge-training { background: #eff6ff; color: #1e40af; border: 1px solid #93c5fd; }

/* Triage badges */
.badge-home_care { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.badge-monitor { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.badge-vet_visit { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
.badge-emergency { background: #fff7ed; color: #9a3412; border: 1px solid #fb923c; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
}

.paw-loader {
    display: flex;
    gap: 6px;
    align-items: center;
}

.paw-dot {
    width: 8px;
    height: 8px;
    background: var(--primary-accent);
    border-radius: 50%;
    animation: bounce 1.2s ease-in-out infinite;
}

.paw-loader span {
    font-size: 12px;
    color: var(--text-light);
    margin-left: 4px;
}

.disclaimer {
    font-size: 11px;
    color: #c4b8ae;
    text-align: center;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

AGENT_META = {
    "health": {
        "label": "🩺 Health",
        "bg": "#fef2f2",
        "color": "#991b1b",
        "border": "#fca5a5",
        "dot": "#ef4444",
    },
    "nutrition": {
        "label": "🥗 Nutrition",
        "bg": "#f0fdf4",
        "color": "#166534",
        "border": "#86efac",
        "dot": "#22c55e",
    },
    "training": {
        "label": "🎓 Training",
        "bg": "#eff6ff",
        "color": "#1e40af",
        "border": "#93c5fd",
        "dot": "#3b82f6",
    },
}

TRIAGE_META = {
    "HOME_CARE": {
        "label": "🏠 Home Care",
        "bg": "#f0fdf4",
        "color": "#166534",
        "border": "#86efac",
        "dot": "#22c55e",
    },
    "MONITOR": {
        "label": "👁 Monitor",
        "bg": "#fffbeb",
        "color": "#92400e",
        "border": "#fcd34d",
        "dot": "#f59e0b",
    },
    "VET_VISIT": {
        "label": "🏥 See a Vet",
        "bg": "#fef2f2",
        "color": "#991b1b",
        "border": "#fca5a5",
        "dot": "#ef4444",
    },
    "EMERGENCY": {
        "label": "🚨 Emergency!",
        "bg": "#fff7ed",
        "color": "#9a3412",
        "border": "#fb923c",
        "dot": "#f97316",
    },
}

def render_badge_html(agents, triage):
    html = '<div class="msg-badges">'
    
    for agent in agents:
        if agent in AGENT_META:
            m = AGENT_META[agent]
            html += (
                f'<span class="badge badge-{agent}" '
                f'style="background:{m["bg"]};color:{m["color"]};border:1px solid {m["border"]}">'
                f'<span class="badge-dot" style="background:{m["dot"]}"></span>'
                f'{m["label"]}</span>'
            )
    
    if triage and triage in TRIAGE_META:
        m = TRIAGE_META[triage]
        html += (
            f'<span class="badge badge-{triage.lower()}" '
            f'style="background:{m["bg"]};color:{m["color"]};border:1px solid {m["border"]}">'
            f'<span class="badge-dot" style="background:{m["dot"]}"></span>'
            f'{m["label"]}</span>'
        )
    
    html += '</div>'
    return html if len(agents) > 0 or triage else ""

def format_llm_output(text: str) -> str:
    import re

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r'\nTRIAGE:\s*\S+[^\n]*', '', text)

    def replace_disclaimer(m):
        body = m.group(1).strip()
        style = (
            "font-size:0.78em; color:#8a7e74; font-style:italic;"
            " border-top:1px solid #e8ddd0; margin-top:12px; padding-top:8px;"
        )
        return f'\n\n<p style="{style}">⚠️ {body}</p>'
    text = re.sub(
        r'\nDISCLAIMER:\s*(.+?)$',
        replace_disclaimer,
        text,
        flags=re.DOTALL,
    )

    text = re.sub(r'\n{3,}', '\n\n', text)

    text = re.sub(
        r'\n((?:(?!DISCLAIMER|TRIAGE)[A-Z][A-Z\s]{2,}:|Phase \d))',
        r'\n\n\1',
        text,
    )
    paragraphs = text.split('\n\n')
    paragraphs = [p.replace('\n', '  \n') for p in paragraphs]
    text = '\n\n'.join(paragraphs)

    return text.strip()


def render_message(role, content, agents=None, triage=None):
    agents = agents or []
    avatar = "🐾" if role == "assistant" else "👤"

    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            badges_html = render_badge_html(agents, triage)
            if badges_html:
                st.markdown(badges_html, unsafe_allow_html=True)
            st.markdown(format_llm_output(content), unsafe_allow_html=True)
        else:
            st.markdown(content)

def render_paw_loader():
   
    st.markdown("""
    <div style="display:flex; gap:10px; align-items:center; padding:12px 16px;">
        <div style="display:flex; gap:6px;">
            <div class="paw-dot" style="animation-delay:0s;"></div>
            <div class="paw-dot" style="animation-delay:0.2s;"></div>
            <div class="paw-dot" style="animation-delay:0.4s;"></div>
        </div>
        <span style="font-size:12px; color:#a3a3a3; margin-left:4px;">
            Agents thinking…
        </span>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your 🐾 **Dog Owner AI Agent**. Set up your dog's profile on the left, "
                "then ask me anything about health, nutrition, or training!"
            ),
            "agents": [],
            "triage": None,
        }
    ]

if "dog_profile" not in st.session_state:
    st.session_state.dog_profile = {
        "name": "Buddy",
        "breed": "Labrador Retriever",
        "age": 2.0,
        "weight": 30.0,
        "sex": "male",
        "neutered": True,
        "allergies": "chicken",
        "healthIssues": "",
        "activityLevel": "high",
    }

with st.sidebar:
    st.markdown(
        '<h1 style="font-family:\'Playfair Display\',serif; font-size:28px; margin:0;">🐾 Dog Agent</h1>',
        unsafe_allow_html=True,
    )
    st.divider()
    
    st.markdown(
        '<h3 style="font-family:\'Playfair Display\',serif; font-size:18px; margin:10px 0 0 0;">🐶 Dog Profile</h3>',
        unsafe_allow_html=True,
    )
    
    dog_name = st.text_input("Name", value=st.session_state.dog_profile["name"])
    dog_breed = st.selectbox(
        "Breed",
        [
            "Labrador Retriever", "Golden Retriever", "German Shepherd", "Bulldog",
            "Poodle", "Beagle", "Rottweiler", "Yorkshire Terrier", "Dachshund",
            "Boxer", "Siberian Husky", "Shih Tzu", "Border Collie", "Mixed Breed",
        ],
        index=0,
    )
    
    col1, col2 = st.columns(2)
    with col1:
        dog_age = st.number_input(
            "Age (years)", 
            min_value=0.0, 
            max_value=25.0, 
            value=st.session_state.dog_profile["age"], 
            step=0.5
        )
    with col2:
        dog_weight = st.number_input(
            "Weight (kg)", 
            min_value=0.5, 
            max_value=100.0, 
            value=st.session_state.dog_profile["weight"], 
            step=0.5
        )
    
    dog_sex = st.radio(
        "Sex",
        ["Male", "Female"],
        index=0 if st.session_state.dog_profile["sex"] == "male" else 1,
        horizontal=True,
    )
    dog_neutered = st.checkbox(
        "Neutered / Spayed",
        value=st.session_state.dog_profile["neutered"],
    )
    dog_activity = st.select_slider(
        "Activity Level",
        options=["Low", "Moderate", "High"],
        value="High" if st.session_state.dog_profile["activityLevel"] == "high" else ("Moderate" if st.session_state.dog_profile["activityLevel"] == "moderate" else "Low"),
    )
    
    dog_allergies = st.text_input(
        "Allergies (comma-separated)",
        value=st.session_state.dog_profile["allergies"],
    )
    dog_health_issues = st.text_input(
        "Health Issues (comma-separated)",
        value=st.session_state.dog_profile["healthIssues"],
    )
    
    st.divider()
    
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat cleared! Ask me anything about your dog. 🐾",
            "agents": [],
            "triage": None,
        }]
        st.rerun()

st.session_state.dog_profile = {
    "name": dog_name,
    "breed": dog_breed,
    "age": dog_age,
    "weight": dog_weight,
    "sex": dog_sex.lower(),
    "neutered": dog_neutered,
    "allergies": dog_allergies,
    "healthIssues": dog_health_issues,
    "activityLevel": dog_activity.lower(),
}

for msg in st.session_state.messages:
    render_message(
        msg["role"],
        msg["content"],
        msg.get("agents", []),
        msg.get("triage"),
    )

if len(st.session_state.messages) == 1:
    st.markdown(
        '<span style="font-size:11px; color:#a3a3a3; text-transform:uppercase; letter-spacing:1px;">Try asking…</span>',
        unsafe_allow_html=True,
    )
    suggestions = [
        "My dog vomited twice and seems tired. Should I be worried?",
        "What should I feed my Labrador with a chicken allergy?",
        "My dog barks excessively when left alone. How do I stop this?",
        "How much should I feed my 4-month-old Golden Retriever puppy?",
        "My dog just ate dark chocolate. What do I do?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": q,
                    "agents": [],
                    "triage": None,
                })
                
                render_message("user", q)

                render_paw_loader()
                
                try:
                    res = requests.post(
                        f"{API_BASE}/chat",
                        json={"question": q, "dog": st.session_state.dog_profile},
                        headers={"ngrok-skip-browser-warning": "true"},
                        timeout=120,
                    )
                    res.raise_for_status()
                    data = res.json()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "agents": data.get("agents", []),
                        "triage": data.get("triage"),
                    })
                    
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Could not reach the backend: **{str(e)}**\n\nMake sure your Flask server is running:\n```\npython app.py\n```",
                        "agents": [],
                        "triage": None,
                    })
                
                st.rerun()

st.divider()

st.markdown(
    '<div class="disclaimer">Not a substitute for professional veterinary advice · Always consult a vet for medical concerns</div>',
    unsafe_allow_html=True,
)

if prompt := st.chat_input(f"Ask about {st.session_state.dog_profile['name']}'s health, nutrition, or training…"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "agents": [],
        "triage": None,
    })
    
    render_message("user", prompt)

    render_paw_loader()
  
    try:
        res = requests.post(
            f"{API_BASE}/chat",
            json={"question": prompt, "dog": st.session_state.dog_profile},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=120,
        )
        res.raise_for_status()
        data = res.json()
     
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "agents": data.get("agents", []),
            "triage": data.get("triage"),
        })
        
    except requests.exceptions.ConnectionError as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ **Cannot connect to the Flask backend.**\n\nMake sure it's running:\n```\npython app.py\n```\n\nError: `{str(e)}`",
            "agents": [],
            "triage": None,
        })
    
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error: **{str(e)}**",
            "agents": [],
            "triage": None,
        })
    
    st.rerun()
