import asyncio
from langgraph.graph import StateGraph
from graph_function import *
from langgraph.graph import END, START
import os
import json
import pandas as pd
from langchain_ollama import ChatOllama
# from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage



# Define the workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)

workflow.add_node("generate", generate)  # generate

workflow.add_node("route_question", route_question)


workflow.add_edge(START,"route_question")
workflow.add_edge("route_question", "retrieve")
workflow.add_edge("retrieve", "generate")

graph = workflow.compile()

async def generate_responses(questions):
    test_schema = []

    # Generate responses for each question asynchronously
    for question in questions:
        inputs = {"question": question}
        result = await graph.invoke(inputs) if asyncio.iscoroutinefunction(graph.invoke) else graph.invoke(inputs)
        generation = result.get("generation") if "generation" in result else None
            # Append the generated result to test_schema
        test_schema.append({
                "question": question,
                "answer": generation,

        })

        return test_schema
        


# if __name__ == "__main__":
#     questions = [
#         "Can you give me a summary of the lecture in week 2"
#     ]

#     results = asyncio.run(generate_responses(questions))
#     for item in results:
#         print(f"Q: {item['question']}")
#         answer = item.get("answer")

#         if answer is not None:
#             response_content = answer.content if hasattr(answer, 'content') else str(answer)
#             print(f"A: {response_content}")
#         else:
#             response_content = "[No response generated]"
#             print("A:", response_content)

#         store_conversations(prompt=item['question'], response=response_content)