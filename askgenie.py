import streamlit as st
import openai
import os
from langdetect import detect

# ------------------ App Configuration ------------------
st.set_page_config(page_title="Bank Genie - Internal Assistant", layout="centered")

# ------------------ Load OpenAI API Key ------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Please set it in your Streamlit Cloud Secrets or local environment.")
    st.stop()
openai.api_key = api_key

# ------------------ Styling ------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f9fa;
    }
    .block-container {
        max-width: 650px;
        background-color: white;
        border-radius: 1.5rem;
        padding: 2rem;
        margin: auto;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 0.75rem;
        padding: 1rem;
        font-size: 1rem;
    }
    .stButton>button {
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
        border: none;
        margin-top: 10px;
        background-color: #000000;
        color: white;
    }
    .custom-answer {
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .example-line {
        margin-top: 1rem;
        font-style: italic;
        color: #333333;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Session Initialization ------------------
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "response" not in st.session_state:
    st.session_state.response = None
if "detail_level" not in st.session_state:
    st.session_state.detail_level = "Short"

# ------------------ Header ------------------
st.title("🏦 Bank Genie - Internal Q&A Assistant")
st.markdown("""
👋 Welcome to **Bank Genie** — Empowering Bank Teams with Instant, Multilingual Support.

💬 Ask any bank-related question below, and Bank Genie will provide accurate, helpful answers tailored to your preference — whether concise or in-depth.

📞 For further assistance or support, feel free to call or WhatsApp us at +91-8830720742.
""")

# ------------------ Detail Level Selector ------------------
st.session_state.detail_level = st.selectbox(
    "Choose answer detail level:",
    ["Short", "Detailed"],
    index=0 if st.session_state.detail_level == "Short" else 1
)

# ------------------ Prompt Template ------------------
BANK_GENIE_PROMPT = """
You are Bank Genie — an internal assistant for bank employees only. You answer only bank-related queries like:
- Account opening/closure, KYC, dormant accounts
- Deposits, withdrawals, cash handling rules
- NEFT, RTGS, UPI, IMPS, cheque handling
- Loans, documentation, eligibility
- Internal tools like Finacle or CBS
- Internal policies, RBI guidelines, audits
- Staff-related questions only if tied to internal policies

❌ Do NOT answer anything unrelated to banking. Respond with:
"I’m designed to answer only internal bank-related questions. Please ask something related to banking."

✅ For valid banking questions:
"""

if st.session_state.detail_level == "Short":
    BANK_GENIE_PROMPT += """
- Give a short, summarized answer (1–3 lines)
- Include 1 simple real-life example (use Indian context and INR)
"""
else:
    BANK_GENIE_PROMPT += """
- Give a clear, helpful answer (up to 5–6 lines)
- Include 1 proper real-life example with Indian context and INR
"""

BANK_GENIE_PROMPT += """
- Keep answer and example on separate lines with space between
- Avoid repeating the word "Example" if it’s already used
- Answer in the same language the user asked
"""

# ------------------ Language Detection ------------------
def detect_user_language(text):
    try:
        text = text.strip()
        if len(text) < 10:
            return "en"
        lang_code = detect(text)
        allowed_languages = {"en", "hi", "mr", "ta", "te", "gu", "kn", "bn", "ml", "pa", "or", "ur", "as", "ne", "si"}
        return lang_code if lang_code in allowed_languages else "en"
    except:
        return "en"

# ------------------ GPT Call ------------------
def get_bank_response(query):
    try:
        query = query.strip()
        if len(query.split()) <= 3 and not query.endswith("?"):
            query = f"What is {query}?"
        user_lang = detect_user_language(query)
        lang_instruction = f"Answer the question in this language: {user_lang}. Use Indian context and INR for all examples. Keep the main answer and example clearly separated with a blank line."
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{BANK_GENIE_PROMPT}\n\n{lang_instruction}"},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"❌ GPT Error: {e}")
        return None

# ------------------ Input Field ------------------
user_input = st.text_input(
    "Ask your question (in any language):",
    value=st.session_state.user_query,
    max_chars=300
)

# ------------------ Ask Button ------------------
if st.button("Ask to Bank Genie") and user_input.strip():
    st.session_state.user_query = user_input
    with st.spinner("Thinking like a banker..."):
        st.session_state.response = get_bank_response(user_input)

# ------------------ Output ------------------
if st.session_state.response:
    reply = st.session_state.response
    if "\n\n" in reply:
        answer, example = reply.split("\n\n", 1)
        example_clean = example.strip().removeprefix("Example:").strip()
        st.markdown(f"""
        <div class='custom-answer'>{answer.strip()}</div>
        <div class='example-line'>💡 Example: {example_clean}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### ✅ Answer\n{reply}")

# ------------------ Footer ------------------
st.markdown("""
---
<div style="text-align:center">
<small>🔐 For internal banking use only | Powered by SuperAI Labs</small>
</div>
""", unsafe_allow_html=True)
