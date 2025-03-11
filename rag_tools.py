import os
import re
import chromadb
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf_files(path):
  pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
  print(f"Found {len(pdf_files)} PDF files in {path}")
  return [f"{path}/{pdf}" for pdf in pdf_files]

def extract_text_from_pdf(pdf_path):
  elements = partition_pdf(filename=pdf_path, strategy="hi_res")
  pdf_text = "\n".join(element.text for element in elements if element.text)
  return pdf_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.replace('\n', ' ')  # Remove newlines
    text = re.sub(r'[^a-zA-Z0-9.,;:\-\'"()\[\] ]', '', text)  # Keep only readable text
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # Remove hyphenations (e.g., "dynam/-ical" -> "dynamical")
    return text.strip()

def split_text_in_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
  text_chunks = text_splitter.split_text(clean_text(text))
  return text_chunks

def vectorize_text(text_chunks):
  vectors = model.encode(text_chunks)
  return vectors

def add_vectors_to_db(file_name,text_chunks, vectors):
  client = chromadb.PersistentClient(path="./chroma_db")
  collection = client.get_or_create_collection("text_chunks")
  for i, vector in enumerate(vectors):
    collection.add(
      ids=[f"{file_name}-{i}"],
      documents=[text_chunks[i]],
      metadatas=[{"file_name": file_name, "index": i}],
      embeddings=[vector]
    )

def query_vector_db(query, max_distance=1.2):
  if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("ChromaDB database not found at ./chroma_db")

  client = chromadb.PersistentClient(path="./chroma_db")
  collection = client.get_or_create_collection("text_chunks")
  query_vector = model.encode([query])[0]
  results = collection.query(query_embeddings=[query_vector], n_results=3)
  
  chunks = []
  related_files = []
  # Filter results based on distance
  for i, distance in enumerate(results['distances'][0]):
      if distance <= max_distance:  # Only include chunks below the threshold
          chunks.append(results['documents'][0][i])
          metadata = results['metadatas'][0][i]
          if metadata['file_name'] not in related_files:
              file_id = metadata['file_name'].split('/')[-1].replace('.pdf', '')
              related_files.append(f"https://arxiv.org/pdf/{file_id}")

  return chunks, related_files
