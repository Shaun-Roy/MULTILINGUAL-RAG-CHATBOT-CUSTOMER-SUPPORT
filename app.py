from flask import Flask, request, jsonify, render_template
import joblib
import langid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load saved logistic regression models and vectorizers
en_intent_model = joblib.load("logreg_model_english.pkl")
hi_intent_model = joblib.load("logreg_model_hindi.pkl")
en_vectorizer = joblib.load("vectorizer_english.pkl")
hi_vectorizer = joblib.load("vectorizer_hindi.pkl")

# Initialize Flask app
app = Flask(__name__)

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "customer-support-chat"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize sentence-transformers for query embedding
en_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
hi_embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"


# English prompt template
ENGLISH_PROMPT = """
You are a helpful customer support assistant.

Based on the following context and intent, provide a clear, concise, step-by-step answer to the user's question. Do not include specific placeholder values like order numbers, URLs, or phone numbers. Just give a clean and helpful response.

Context:
{context}

Intent:
{intent}

Question:
{question}

Answer:
"""



#hindi prompt template
HINDI_PROMPT = """
आप एक सहायक ग्राहक सेवा सहायक हैं।

नीचे दिए गए संदर्भ और उद्देश्य के आधार पर उपयोगकर्ता के प्रश्न का स्पष्ट, संक्षिप्त और चरण-दर-चरण उत्तर दें। कृपया ऑर्डर नंबर, वेबसाइट URL या फ़ोन नंबर जैसे प्लेसहोल्डर शामिल न करें। उत्तर सीधे और मददगार होना चाहिए।

संदर्भ:
{context}

उद्देश्य:
{intent}

प्रश्न:
{question}

उत्तर:
"""



import langid

def detect_language(text):
    lang, _ = langid.classify(text)
    if lang == "hi":
        return "hindi"
    else:
        return "english"


def predict_intent(text, language):
    if language == "english":
        X = en_vectorizer.transform([text])
        return en_intent_model.predict(X)[0]
    else:
        X = hi_vectorizer.transform([text])
        return hi_intent_model.predict(X)[0]

def retrieve_context(query, language, top_k=5):
    model = en_embed_model if language == "english" else hi_embed_model
    query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()
    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"language": language}
    )
    if res['matches']:
        return "\n\n".join([match["metadata"]["text"] for match in res['matches']])
    else:
        return ""

def call_sarvam_llm(prompt):
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sarvam-m",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(SARVAM_CHAT_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        return "Sorry, I couldn't generate a response at this time."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    print("\n" + "=" * 60)
    print(f"🔎 User Query: {user_query}")

    language = detect_language(user_query)
    print(f"🗣️ Detected Language: {language}")

    intent = predict_intent(user_query, language)
    print(f"📌 Predicted Intent: {intent}")

    context = retrieve_context(user_query, language)
    print(f"📂 Retrieved Context (first 1000 chars):\n{context[:1000]}")

    # Construct prompt
    prompt = ENGLISH_PROMPT.format(
        context=context, intent=intent, question=user_query
    ) if language == "english" else HINDI_PROMPT.format(
        context=context, intent=intent, question=user_query
    )

    answer = call_sarvam_llm(prompt)
    print(f"🤖 LLM Response:\n{answer}\n")
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
