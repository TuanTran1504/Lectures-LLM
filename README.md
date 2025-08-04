# 🎓 Transcript-RAG: Lecture Question Answering System

This project enables students to upload lecture recordings or transcripts and ask questions based on the content. It leverages a Retrieval-Augmented Generation (RAG) pipeline powered by LLMs (via Ollama), PostgreSQL, and ChromaDB for document retrieval and semantic chunking.

---

## 🚀 Features

- Upload lecture files (`.txt`, `.pdf`, `.mp4`, `.mp3`)
- Transcribe and chunk content semantically using `nltk` + `sentence-transformers`
- Store structured content into PostgreSQL
- Ask questions and get context-aware answers from LLMs (via Ollama)
- GPU and CPU versions supported via Docker Compose

---

## 🛠️ Tech Stack

| Component      | Tool/Library                                 |
|----------------|----------------------------------------------|
| Backend        | Flask, Python                                |
| Vector DB      | ChromaDB                                     |
| Embedding      | `nomic-embed-text`, `sentence-transformers`  |
| Language Model | Ollama with `llama3:latest`                  |
| Database       | PostgreSQL                                   |
| Containerization | Docker, Docker Compose                    |

---

## 🧱 Project Structure

```
project/
├── app.py                   # Flask web application
├── main.py                 # Semantic chunking and DB storage
├── transcript.py           # Transcript reading/parsing logic
├── templates/
│   ├── upload.html         # Upload form
│   └── ask.html            # Question answering UI
├── uploads/                # Uploaded user lecture files
├── init/                   # PostgreSQL initialization scripts
├── Dockerfile              # App Dockerfile
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 🐳 Docker Compose

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/transcript-rag.git
   cd transcript-rag
   ```

2. Start the stack:
   ```bash
   docker compose up --build
   ```

3. Access the web app:
   ```
   http://localhost:5000
   ```

### 🧠 Pull the LLM (Ollama)

Before starting, make sure to pull the required model:

```bash
docker exec -it ollama-cpu ollama pull llama3:latest
```

---

## 📄 Upload & Ask Flow

1. Go to the upload page (`/`)
2. Upload a `.txt`, `.pdf`, `.mp3`, or `.mp4` file and select a lecture week
3. After processing, you're redirected to the `/ask` page
4. Ask questions, and get context-aware answers based on the transcript

---

## 🧠 Models

- **LLM**: `llama3:latest` via [Ollama](https://ollama.com)
- **Embedding Model**: `nomic-embed-text` or `sentence-transformers/all-MiniLM-L6-v2`
- **Chunking**: NLTK-based sentence tokenizer + semantic similarity

---

## 📝 Environment Variables

These can be defined in `docker-compose.yml`:

```yaml
POSTGRES_DB=memmory_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=150402
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
OLLAMA_NO_CUDA=1
```

---

## 🧪 Dev Tips

- Use `OLLAMA_NO_CUDA=1` to disable GPU in Ollama
- Mount a volume like `./uploads:/app/uploads` to persist uploaded files
- All logs (Flask + PostgreSQL) will appear in your Docker terminal for easy debugging

---

## 📦 TODOs

- [ ] Add authentication
- [ ] Add transcript summarization
- [ ] Add search and history features for questions
- [ ] Optional support for Whisper for mp3/mp4 transcription

---

## 📄 License

MIT License © 2025 [Your Name or Org]
