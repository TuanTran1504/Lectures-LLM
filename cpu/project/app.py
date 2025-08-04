from flask import Flask, request, render_template, redirect, url_for
import os
import psycopg2
from werkzeug.utils import secure_filename
from transcript import *
from main import *
from database import *
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/ask', methods=['GET', 'POST'])
def ask():
 
    answer = None
    question = None

    if request.method == 'POST':
        question = request.form.get('question')
        results = asyncio.run(generate_responses([question]))
        for result in results:
            answer_text = result.get("answer")
            if answer_text:
                answer = (
                    answer_text.content if hasattr(answer_text, 'content')
                    else str(answer_text)
                )
                break

    return render_template('ask.html', question=question, answer=answer, success=True)
@app.route('/upload', methods=['POST'])
def upload_file():
    week = request.form.get('week')
    file = request.files.get('file')

    print(f"📥 Received upload for week: {week}")
    
    if not file or file.filename == '':
        print("❌ No file selected")
        return "No file selected", 400

    if not allowed_file(file.filename):
        print(f"❌ File type not allowed: {file.filename}")
        return "File type not allowed", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    print(f"📂 Saving file to: {filepath}")
    file.save(filepath)

    print("📤 Starting processing pipeline...")
    upload(week_number=week, transcript_path=filepath)

    print("✅ Upload and processing complete.")
    return redirect(url_for('ask', week=week,success=True))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)