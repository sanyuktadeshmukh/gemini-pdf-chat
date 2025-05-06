from flask import Flask, request, render_template, redirect, url_for
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, just say "Answer is not available in the context." Don't guess.

    ```
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    models = genai.list_models()
    print("\nAvailable Models and Supported Methods:\n")
    for model in models:
        print(f"Model Name: {model.name}")
        print(f"  Supported Generation Methods: {model.supported_generation_methods}")
        print("-" * 50)
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route("/", methods=["GET", "POST"])
def home():
    message = None
    error = None

    if request.method == "POST" and "pdf_files" in request.files:
        files = request.files.getlist("pdf_files")
        file_paths = []

        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            file_paths.append(file_path)

        try:
            raw_text = get_pdf_text(file_paths)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            message = "PDFs processed successfully."
        except Exception as e:
            error = f"An error occurred while processing PDFs: {str(e)}"

    return render_template("index.html", success=message, error=error)

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.form.get("question", "").strip()
    if not user_question:
        return render_template("index.html", error="Please provide a question.")

    index_path = "faiss_index"
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        return render_template("index.html", error="No PDF files have been processed yet. Please upload and process PDFs first.")

    try:
        answer = process_question(user_question)
        return render_template("index.html", answer=answer, question=user_question)
    except Exception as e:
        return render_template("index.html", error=f"An error occurred while answering the question: {str(e)}")

@app.route("/delete", methods=["POST"])
def clear():
    try:
        # Clear uploaded PDFs
        for file in os.listdir("uploaded_pdfs"):
            file_path = os.path.join("uploaded_pdfs", file)
            os.remove(file_path)

        # Clear the FAISS index folder
        faiss_index_path = "faiss_index"
        if os.path.exists(faiss_index_path):
            for file in os.listdir(faiss_index_path):
                file_path = os.path.join(faiss_index_path, file)
                os.remove(file_path)

        return redirect(url_for("home"))
    except Exception as e:
        error_message = f"An error occurred while clearing the files: {str(e)}"
        return render_template("index.html", error=error_message)


if __name__ == "__main__":
    app.run(debug=True)
