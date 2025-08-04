# Wait for Ollama to be ready
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama API..."
  sleep 2
done


echo "PostgreSQL is ready. Starting Flask app..."
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000