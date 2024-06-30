import os
import PyPDF2
from time import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory containing the PDF files
PDF_DIRECTORY = 'pdfs'
# Directory to store Chroma embeddings
EMBEDDINGS_DIRECTORY = 'embeddings'

# Ensure embeddings directory exists
os.makedirs(EMBEDDINGS_DIRECTORY, exist_ok=True)

def extract_pdf_text(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text

def create_vector_store(texts, metadatas, persist_directory):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
    docsearch.persist()
    return docsearch

def load_chroma_data(persist_directory, embeddings_model):
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
    return docsearch

def load_or_create_embeddings(pdf_file, pdf_text):
    pdf_name = os.path.splitext(pdf_file)[0]
    embeddings_path = os.path.join(EMBEDDINGS_DIRECTORY, pdf_name + '.pdf')

    if os.path.exists(embeddings_path):
        print(f"Embeddings already exist for {pdf_file}. Loading from file...")
        embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        docsearch = load_chroma_data(embeddings_path, embeddings_model)
    else:
        print(f"Creating embeddings for {pdf_file}...")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        texts = text_splitter.split_text(pdf_text)

        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        try:
            # Create a Chroma vector store and persist it
            docsearch = create_vector_store(texts, metadatas, embeddings_path)
            print(f"Embeddings stored in {embeddings_path}")
        except Exception as e:
            print(f"Error creating Chroma vector store: {e}")
            raise e

    return docsearch

def process_pdfs():
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_file_path = os.path.join(PDF_DIRECTORY, pdf_file)
        
        print(f"Processing {pdf_file}...")

        # Measure time taken to read the PDF file
        start_time = time()
        pdf_text = extract_pdf_text(pdf_file_path)
        end_time = time()
        print(f"Time taken to read PDF: {end_time - start_time} seconds")

        # Load or create embeddings
        load_or_create_embeddings(pdf_file, pdf_text)

if __name__ == "__main__":
    process_pdfs()
