import chromadb
import psycopg
import ollama
import ast
from psycopg.rows import dict_row
import os
ollama_client = ollama.Client(host=os.environ.get("OLLAMA_API_URL", "http://ollama:11434"))

def connect_db(param=None):
    if param is None:
        param = {
            'dbname': os.environ.get('POSTGRES_DB', 'memmory_agent'),
            'user': os.environ.get('POSTGRES_USER', 'postgres'),
            'password': os.environ.get('POSTGRES_PASSWORD', '150402'),
            'host': os.environ.get('POSTGRES_HOST', 'postgres'),  # <--- Important: use container name
            'port': os.environ.get('POSTGRES_PORT', '5432')
        }
    return psycopg.connect(**param)

def fetch_lectures(week):
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM lectures WHERE week = %s", (week,))
        lectures = cursor.fetchall()
    conn.close()
    return lectures
def create_lecture_vdb(lectures, client):
    vector_db_name = "lectures"
    try:
        client.delete_collection(name=vector_db_name)
    except chromadb.errors.NotFoundError:
        print(f"Collection '{vector_db_name}' not found. Skipping deletion.")
     
    vector_db = client.create_collection(name=vector_db_name)

    for c in lectures:
        serialized_doc = f"Week {c['week']} | Chunk {c['chunk_id']}:\n{c['content']}"
        response = ollama_client.embeddings(model='nomic-embed-text', prompt=serialized_doc)
        embedding = response['embedding']
        
        vector_db.add(
            ids=[f"{c['week']}_{c['chunk_id']}"],
            embeddings=[embedding],
            documents=[serialized_doc]
        )

def classify_embedding(query, context):
    classify_msg=(
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text.'
        "You will not respond as an AI assistant. You only respond 'yes' or 'no'."
        "Determine whether the context contains data that directly related to the search query"
        ". If the context is seemingly exactly what the search query needs, response 'yes'. If it is anything"
        "but directly related response 'no'. Do not respond 'yes' unless the content is highly relevant to the search query."    
    )
    classify_covo = [
    {'role': 'system', 'content': classify_msg},

    # Example 1 - positive
    {'role': 'user', 'content': 'SEARCH QUERY: What is the user\'s name?\n\nEMBEDDED CONTEXT: You are Dustin'},
    {'role': 'assistant', 'content': 'yes'},

    # Example 2 - negative
    {'role': 'user', 'content': 'SEARCH QUERY: What is the user\'s name?\n\nEMBEDDED CONTEXT: The user lives in Sydney'},
    {'role': 'assistant', 'content': 'no'},

    # Example 3 - positive
    {'role': 'user', 'content': 'SEARCH QUERY: What company does the user work at?\n\nEMBEDDED CONTEXT: The user works at OpenAI as a researcher.'},
    {'role': 'assistant', 'content': 'yes'},

    # Example 4 - negative
    {'role': 'user', 'content': 'SEARCH QUERY: What company does the user work at?\n\nEMBEDDED CONTEXT: The user enjoys hiking on weekends.'},
    {'role': 'assistant', 'content': 'no'},

    # Example 5 - ambiguous (but should lean negative)
    {'role': 'user', 'content': 'SEARCH QUERY: What is the user\'s favorite food?\n\nEMBEDDED CONTEXT: The user went to Italy last summer.'},
    {'role': 'assistant', 'content': 'no'},

    # Example 6 - positive
    {'role': 'user', 'content': 'SEARCH QUERY: What is the user\'s favorite food?\n\nEMBEDDED CONTEXT: The user\'s favorite food is sushi.'},
    {'role': 'assistant', 'content': 'yes'},
    {'role':'user','content':f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    response = ollama_client.chat(model='llama3', messages=classify_covo)
    return response['message']['content'].strip().lower()
def create_queries_lecture(prompt):
    query_msg = (
        "You are a first principle reasoning search query AI agent."
        "Your list of search queries will be ran on an embedding database of all the lectures"
        ".With first principles create a Python list of queries to"
        "search the embeddings database for any data that would be necessary to have access to in"
        "order to correctly respond to the prompt. Your response must be a Python list with no syntax errors."
        "Do not explain anything and do not ever generate anything but a perfect syntax Python list"
    )
    query_convo = [
        {'role':"system", 'content': query_msg},
        {'role':'user', 'content':'Write an email to my car insurance company and create a persuasive request for them to lower my monthly rate.'},
        {'role':'assistant','content':'["What is the users name?", "What is the users current auto insurance provider", "What is the monthly rate the user currently pays for auto insurance?"]'},
        {'role':'user','content':'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3'},
        {'role':'assistant','content':'["Llama3 voice assistant","Python voice assistant","OpenAI TTS","Openai speak"]'},
        {'role':'user','content': prompt}
    ]
    response = ollama_client.chat(model='llama3', messages=query_convo)
    print(f"\nVector database queries: {response['message']['content']} \n")
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]
    
def fetch_conversations(limit=100):
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT %s",(limit,))
        conversations = cursor.fetchall()
    conn.close()
    return conversations

def fecth_working_memory(client,query):
    embeddings=set()
    conversations= fetch_conversations(limit=10)
    create_vector_db(conversations, client)
    response = ollama_client.embeddings(model='nomic-embed-text', prompt=query)
    query_embedding = response['embedding']
    vector_db = client.get_collection(name="conversations")
    results= vector_db.query(query_embeddings=[query_embedding],n_results=10)
    best_embeddings = results['documents'][0]
    for best in best_embeddings:
        embeddings.add(best)
    return embeddings

def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
            (prompt, response)
        )
        conn.commit()
    conn.close()
def create_vector_db(conversations, client):
    vector_db_name="conversations"
    try:
        client.delete_collection(name=vector_db_name)
    except chromadb.errors.NotFoundError:
        print(f"Collection '{vector_db_name}' not found. Skipping deletion.")
     
    vector_db=client.create_collection(name=vector_db_name)
    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} / response: {c['response']}"
        response= ollama_client.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']
        
        vector_db.add(
            ids=[str(c['id'])],
            embeddings = [embedding],
            documents = [serialized_convo]
        )
def create_queries(prompt):
    query_msg = (
        "You are a first principle reasoning search query AI agent."
        "Your list of search queries will be ran on an embedding database of all your conversations"
        "you have ever had with the user. With first principles create a Python list of queries to"
        "search the embeddings database for any data that would be necessary to have access to in"
        "order to correctly respond to the prompt. Your response must be a Python list with no syntax errors."
        "Do not explain anything and do not ever generate anything but a perfect syntax Python list"
    )
    query_convo = [
        {'role':"system", 'content': query_msg},
        {'role':'user', 'content':'Write an email to my car insurance company and create a persuasive request for them to lower my monthly rate.'},
        {'role':'assistant','content':'["What is the users name?", "What is the users current auto insurance provider", "What is the monthly rate the user currently pays for auto insurance?"]'},
        {'role':'user','content':'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3'},
        {'role':'assistant','content':'["Llama3 voice assistant","Python voice assistant","OpenAI TTS","Openai speak"]'},
        {'role':'user','content': prompt}
    ]
    response = ollama_client.chat(model='llama3', messages=query_convo)
    print(f"\nVector database queries: {response['message']['content']} \n")
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]
def retrieve_embeddings(queries, client, results_per_query=2, db="conversations"):
    embeddings=set()
    for query in queries:
        response = ollama_client.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']
        vector_db = client.get_collection(name=db)
        results= vector_db.query(query_embeddings=[query_embedding],n_results=results_per_query)
        best_embeddings = results['documents'][0]
        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query, best):
                    embeddings.add(best)
    return embeddings


