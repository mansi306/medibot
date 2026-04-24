import os
import warnings
from dotenv import load_dotenv
import speech_recognition as sr
from googletrans import Translator
from langdetect import detect
from gtts import gTTS
import playsound
import datetime
import tempfile

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment
load_dotenv()
warnings.filterwarnings("ignore")

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_ENDPOINT_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

if not HF_TOKEN:
    raise ValueError("⚠ Set HF_TOKEN environment variable first!")

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """[INST]
<<SYS>>
You are a clinical assistant. Use ONLY the provided medical context to answer.
If unsure, respond: "This requires consultation with a medical professional".
<</SYS>>

Context: {context}
Question: {question}

Provide a concise, evidence-based answer using medical terms from the context: [/INST]
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    raise RuntimeError(f"❌ FAISS load failed: {str(e)}\nDid you run create_memory_for_llm.py first?")

def load_llm():
    return HuggingFaceEndpoint(
        endpoint_url=HUGGINGFACE_ENDPOINT_URL,
        temperature=0.2,
        max_new_tokens=512,
        repetition_penalty=1.25,
        huggingfacehub_api_token=HF_TOKEN
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    input_key="question",
    chain_type_kwargs={"prompt": set_custom_prompt()}
)

# 🎤 Voice Input
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙 Speak your medical query:")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# 🔊 gTTS Text-to-Speech
def speak(text, mute=False, lang="en"):
    if mute or not text:
        return
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_path)
        playsound.playsound(temp_path)
        os.remove(temp_path)  # clean up
    except Exception as e:
        print(f"🔊 TTS Error: {str(e)}")

# 🌐 Translate answer
def translate_answer(answer, target_language_code):
    translator = Translator()
    translated = translator.translate(answer, dest=target_language_code)
    return translated.text

# 🌍 Auto detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# 🔄 Choose translation language
def choose_language(auto_detected_lang):
    print("\n🌍 Choose output language:")
    print("1. English (default)")
    print("2. Hindi")
    print("3. Gujarati")
    print(f"4. Auto (same as input: {auto_detected_lang})")
    choice = input("Enter 1/2/3/4: ").strip()

    lang_map = {
        "1": "en",
        "2": "hi",
        "3": "gu",
        "4": auto_detected_lang
    }
    return lang_map.get(choice, "en")

# 💾 Save Q&A history
def save_history(question, answer, lang_code):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("qa_history.txt", "a", encoding="utf-8") as file:
        file.write(f"\n--- {timestamp} ---\n")
        file.write(f"Q [{lang_code}]: {question}\n")
        file.write(f"A [{lang_code}]: {answer}\n")

# 🚀 Main
if __name__ == "__main__":
    try:
        mute_audio = input("\n🔈 Mute audio responses? (y/n): ").strip().lower() == 'y'

        mode = input("\n🎧 Press V for voice or T to type your medical question (V/T): ").strip().lower()
        if mode == 'v':
            user_query = voice_input()
            if not user_query:
                raise ValueError("Couldn't understand your voice. Try typing instead.")
        else:
            user_query = input("\n🔍 Medical Query: ").strip()

        if not user_query:
            raise ValueError("Please enter a medical question.")

        print(f"\n📨 You asked: {user_query}")

        # Call LLM
        response = qa_chain.invoke({"question": user_query})
        answer = response["result"].strip()

        # Detect language
        detected_lang = detect_language(user_query)

        # Choose output language
        target_lang = choose_language(detected_lang)
        if target_lang != "en":
            answer = translate_answer(answer, target_lang)

        print("\n📝 Answer:")
        print(answer)

        # Speak out answer
        hear = input("\n🔊 Do you want to hear the answer? (y/n): ").strip().lower()
        if hear == 'y':
            speak(answer, mute=mute_audio, lang=target_lang)

        # Show source documents
        print("\n📚 Source References:")
        sources = {}
        for doc in response["source_documents"]:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "N/A")
            if source not in sources:
                sources[source] = []
            sources[source].append(str(page))

        for source, pages in sources.items():
            print(f"- {source} (Pages: {', '.join(pages)})")

        # Save to file
        save = input("\n💾 Save this Q&A to history file? (y/n): ").strip().lower()
        if save == 'y':
            save_history(user_query, answer, target_lang)
            print("✅ History saved to qa_history.txt")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
