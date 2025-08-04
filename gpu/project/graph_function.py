import operator
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langchain.schema import Document
from langgraph.graph import END
import json
import torch
import getpass
import os
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
# from langchain_community.tools.tavily_search import TavilySearchResults
import os
from database import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_json_mode = ChatOllama(model="llama3:latest", temperature=0, format="json", base_url=os.environ.get("OLLAMA_API_URL", "http://ollama:11434"))

llm = ChatOllama(model="llama3:latest", temperature=0,base_url=os.environ.get("OLLAMA_API_URL", "http://ollama:11434"))
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    convo: list 
    context: list
    data_path: str #Data path
    question: str  # User question
    generation: str  # LLM generation
    answers: int  # Number of answers generated


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    prompt = state["question"]
    convo=[]
    context=[]
    content_parts = []
    client = chromadb.Client()

    path = state.get("data_path")
    lectures_embeddings = []
    if path: 
        week_number = int(path.replace("week", "").strip())
        lectures = fetch_lectures(week=week_number)
        create_lecture_vdb(lectures=lectures, client=client)
        lectures_queries = create_queries_lecture(prompt=prompt)
        lectures_embeddings = retrieve_embeddings(lectures_queries, db="lectures", results_per_query=20, client=client)
    
    conversations = fetch_conversations()
    create_vector_db(conversations, client=client)
    memory_queries = create_queries(prompt=prompt)
    memory_embeddings = retrieve_embeddings(memory_queries, client=client)
    working_memory=fecth_working_memory(client=client, query=prompt)

    memory_embeddings.update(working_memory)
    
    if lectures_embeddings:
        lecture_context = "\n".join(lectures_embeddings)
        context.append({
            "content": f"Context:\n{lecture_context}"
        })

    for memory in memory_embeddings:
        convo.append({'role': 'user', 'content': f'Previously you have this conversation: query: {memory}'})

    convo.append({
    'role': 'user',
    'content': prompt
    })

    return {'convo': convo, 'context':context}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    convo = state.get("convo", []) 
    context = state.get("context", [])
    
    rag_prompt = """You are an assistant for question-answering tasks related to any concern
    You have the context of previous conversation and you have to use it to answer the question
    Answer the query in a clear, accurate, and concise manner.
    Here is the memory of the previous to use to answer the question:

    {memory} 

    Now, think carefully about the context

    {context}

    Now, review the user question:

    {question}

    Provide an answer to this questions using the above context and memory if neccesary
    Try to give the answer that the user need.
    Responses should be short and direct, delivering the necessary information without filler.
    If the question is broad, please provide a more general answer

    Answer:"""

    rag_prompt_formatted = rag_prompt.format(memory=convo, question=question, context=context)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {'generation': generation}


def route_question(state):
    print("---ROUTE QUESTION---")
    router_instructions = """You are an expert at routing a user question to a vectorstore or not.

    The vectorstore contains documents related to lectures of every week of a subject

    Return JSON with a single key "datasource", which is 'week' + number (e.g., 'week3') depending on the query.
    If there is no week mentioned in the query, do not return anything.
    """

    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )

    try:
        response = json.loads(route_question.content)
        source = response.get("datasource", None)
    except json.JSONDecodeError as e:
        print("JSON parse error:", e)
        source = None
    return {"data_path": source}
    

