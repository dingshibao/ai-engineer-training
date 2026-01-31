from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
import os
import asyncio
import json

def company_recognize(query: str) -> list[str]:
    """
    公司实体识别
    """

    llm = OpenAILike(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com/v1",
        is_chat_model=True,  # 使用 /chat/completions 接口
    )

    prompt_template = get_prompt_template()
    messages = prompt_template.format_messages(query_str=query)
    llm_response = llm.chat(messages=messages)

    company_names = None
    try:
        company_names = json.loads(llm_response.message.content.strip())
        print("公司实体识别结果：", company_names)
    except json.JSONDecodeError:
        print("公司实体识别结果格式错误，返回空列表")
        company_names = []
    return company_names

def get_vector_store():
    """
        获取或创建 Milvus 向量存储
    """
    try:
        # 尝试获取运行中的事件循环
        loop = asyncio.get_running_loop()
        # 如果有运行的事件循环，可以直接创建（nest_asyncio 已应用）
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name="graph_rag_companies",
            overwrite=False,
            similarity_metric="COSINE",
        )
    except RuntimeError:
        # 如果没有运行的事件循环，创建新的事件循环并在其中创建
        async def _create_store_async():
            """在异步函数中创建向量存储（确保事件循环运行）"""
            return MilvusVectorStore(
                uri="http://localhost:19530",
                collection_name="graph_rag_companies",
                overwrite=False,
                similarity_metric="COSINE",
            )
        vector_store = asyncio.run(_create_store_async())
        
    return vector_store

def get_prompt_template():
    """
    获取公司实体识别的提示词模板
    """
    # 增加提示词模版，
    return ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""你是一个公司实体识别专家，请根据用户的问题，返回出现的公司实体名称列表。
                返回格式：["公司实体名称1", "公司实体名称2", ...]"""
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="""用户问题：{query_str}, 请返回出现的公司实体名称列表。"""
            )
        ]
    )