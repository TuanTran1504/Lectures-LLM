import psycopg2
from psycopg2.extras import execute_batch
import fitz  # PyMuPDF
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, util
import whisper
import ffmpeg
import os
import tempfile
import torch
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
    if ext in [".mp3", ".mp4", ".wav"]:
        return transcribe_fast(filepath)
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

def speed_up_audio(input_path, speed=2.0):
    # Validate speed
    if speed <= 0 or speed > 4.0:
        raise ValueError("Speed must be between 0 and 4.0")

    # Temp output path
    base, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(tempfile.gettempdir(), f"{base}_fast.wav")

    # Apply audio speed filter
    # Note: You can chain atempo filters up to 2x per filter
    filters = []
    remaining_speed = speed
    while remaining_speed > 2.0:
        filters.append("atempo=2.0")
        remaining_speed /= 2.0
    filters.append(f"atempo={remaining_speed}")
    atempo_filter = ",".join(filters)

    ffmpeg.input(input_path).output(
        output_path,
        format='wav',
        acodec='pcm_s16le',
        ac=1,
        ar='16000',
        **{'filter:a': atempo_filter}
    ).overwrite_output().run(quiet=True)

    return output_path

def transcribe_with_whisper(audio_path, model_size="base"):
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size).to("cuda")
    print("Transcribing...")
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_fast(input_path, speed=2.0, model_size="base"):
    print(f"Speeding up file: {input_path}")
    sped_up_path = speed_up_audio(input_path, speed)
    print(f"Temporary sped-up file at: {sped_up_path}")
    text = transcribe_with_whisper(sped_up_path, model_size)
    os.remove(sped_up_path)  # Cleanup temp file
    return text
