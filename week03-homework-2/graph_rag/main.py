
from fastapi import FastAPI, Body
import uvicorn

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
import os
from pathlib import Path
from graph_rag_retriever import multi_hop_query
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    return QueryResponse(answer=multi_hop_query(query=request.query))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)