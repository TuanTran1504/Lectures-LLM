import psycopg2
from psycopg2.extras import execute_batch
import fitz  # PyMuPDF
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, util

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")
llm = ChatOllama(model="llama3:latest", temperature=0, base_url=os.environ.get("OLLAMA_API_URL", "http://ollama:11434"))
def read_transcript(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    elif ext == '.pdf':
        return extract_text_from_pdf(filepath)
    else:
        raise ValueError("Unsupported file type. Please use .txt or .pdf")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
def semantic_chunking(text, max_words=500, min_words=200, similarity_threshold=0.4):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_embeddings = [embeddings[0]]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        emb = embeddings[i]
        
        avg_sim = util.cos_sim(emb, current_embeddings).mean().item()
        current_len = sum(len(s.split()) for s in current_chunk)

        if (avg_sim < similarity_threshold and current_len >= min_words) or current_len >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_embeddings = [emb]
        else:
            current_chunk.append(sentence)
            current_embeddings.append(emb)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
def generate_chunk_topic(chunk, llm):
    prompt = f"""
    Summarize the following lecture transcript chunk into a short topic title (max 8 words):

    "{chunk}"
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def store_chunks_to_postgres(chunks, week, llm):
    try:
        conn = psycopg2.connect(
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=os.environ["POSTGRES_PORT"]
        )
        cursor = conn.cursor()

        data_to_insert = []
        for idx, chunk in enumerate(chunks):
            topic = generate_chunk_topic(chunk, llm)
            data_to_insert.append((week, idx, chunk, topic))

        execute_batch(
            cursor,
            "INSERT INTO lectures (week, chunk_id, content, topic) VALUES (%s, %s, %s, %s)",
            data_to_insert
        )
        conn.commit()
        print(f"✅ Stored {len(chunks)} chunks with topics into DB for week {week}.")

    except Exception as e:
        print(f"❌ Database error: {e}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# ---- Run the workflow ----
def upload(week_number, transcript_path):
    print(f"📦 Received file for week {week_number}: {transcript_path}", flush=True)
    try:
        full_text = read_transcript(transcript_path)
        print(f"📄 Transcript length: {len(full_text)} chars", flush=True)

        chunks = semantic_chunking(full_text, max_words=500)
        print(f"🧩 Created {len(chunks)} chunks", flush=True)

        store_chunks_to_postgres(chunks, week=week_number, llm=llm)
        print("✅ Successfully stored to DB", flush=True)
    except Exception as e:
        print(f"❌ Error in upload pipeline: {e}", flush=True)