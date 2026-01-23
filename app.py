import os
import gradio as gr
from transformers import pipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# CONFIGURATION
PDF_PATH = "Employee_HB_Final.pdf"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
TOP_K = 2


# LOAD AND PROCESS DOCUMENT
def load_documents():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found.")
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def load_vectorstore(chunks):
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    path = os.path.join(VECTORSTORE_DIR, "faiss_index")

    if os.path.exists(path):
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(path)
    return vectorstore


def load_llm():
    return pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_new_tokens=200,
        temperature=0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )


# RAG ANSWERING LOGIC
def answer_question(question):
    try:
        docs = vectorstore.similarity_search(question, k=TOP_K)

        if not docs:
            return "I don’t know based on the provided document."

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a professional, supportive workplace assistant.
Your tone should be clear, respectful, and friendly.

Answer the question strictly using the context below.
Summarize the relevant points clearly and comprehensively.
Do not repeat the same sentence or phrase.
Do NOT add advice or information not explicitly stated.
If the answer is not contained in the context, say exactly:
"I don’t know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
        response = llm(prompt)[0]["generated_text"].strip()
        return response or "I don’t know based on the provided document."

    except Exception as e:
        print(f"Chat error: {e}")
        return "Sorry — something went wrong while processing your question. Please try again."


# UI STATE MANAGEMENT
chat_history = []


def chat_ui(user_input):
    reply = answer_question(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})
    return chat_history, ""  # auto-clear textbox


def clear_chat():
    chat_history.clear()
    return []


# APPLICATION STARTUP
try:
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = load_vectorstore(chunks)
    llm = load_llm()
except Exception as e:
    raise SystemExit(f"Startup failed: {e}")


# GRADIO UI
with gr.Blocks(css="""
#chatbot {
    height: 520px;
    overflow-y: auto;
}
""") as demo:

    gr.Markdown("""
    <h1 style="text-align:center;">🏢 PeopleLoop Remote Work Assistant</h1>
    <p style="text-align:center;">
    This assistant answers questions <b>strictly based on the Remote Employee Handbook</b>.
    </p>
    """)

    chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Type your question here...",
            show_label=False,
            scale=4
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Chat")

    send_btn.click(chat_ui, textbox, [chatbot, textbox])
    textbox.submit(chat_ui, textbox, [chatbot, textbox])
    clear_btn.click(clear_chat, None, chatbot)

demo.launch()
