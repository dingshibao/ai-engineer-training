from neo4j_manager import Neo4jManager
from llama_index.core import StorageContext
from llama_index.llms.openai_like import OpenAILike
import os
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole, PromptTemplate, PromptType
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from milvus_manager import MilvusManager
from company_recognize import company_recognize


# KnowledgeGraphQueryEngine 需要 graph_query_synthesis_prompt 将自然语言转为 Cypher
GRAPH_QUERY_SYNTHESIS_PROMPT = PromptTemplate(
    """
    任务：生成一个用于查询图数据库的 Cypher 语句。
    目标：根据给定的实体名称列表，查出与这些实体有**直接关系**（一度关系）的所有关系及其属性，以及相邻节点的属性。只做一度扩展，不要多跳。
    说明：
    - 仅使用架构中提供的关系类型和属性，不要使用未提供的。
    - 实体名称来自问题中的列表，需在 Cypher 中用该列表匹配节点（如 WHERE n.name IN 列表）。
    - 返回直接相连的关系类型、关系上的所有属性、以及另一侧节点的属性。
    架构：{schema}
    注意：回答中只输出 Cypher 语句，不要包含解释或道歉。
    问题（实体名称列表）：{query_str}
    """,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True,
)

def cypher_query(entity_name_list: list[str]):
    """
    执行 Cypher 查询
    """

    llm = OpenAILike(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com/v1",
        is_chat_model=True,
    )
    graph_store = Neo4jManager(
        uri=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    ).get_graph_store()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)


    prompt_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="你是一个专业的知识图谱查询专家，请根据实体名称列表，返回最相关的信息。"
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"实体名称列表：{entity_name_list}"
            )
        ]
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        refine_template=prompt_template,
    )

    query_engine = KnowledgeGraphQueryEngine(
        storage_context=storage_context,
        llm=llm,
        graph_query_synthesis_prompt=GRAPH_QUERY_SYNTHESIS_PROMPT,
        verbose=True,  # 打印生成的 Cypher 查询，便于调试
        response_synthesizer=response_synthesizer,
    )

    query_str =', '.join(entity_name_list)
    return query_engine.query(query_str).response


def rag_query(entity_name_list: list[str]):
    """
    执行 RAG 查询
    """
    vector_store = MilvusManager(uri="http://localhost:19530", collection_name="graph_rag_companies", overwrite=False, embedding_model=Settings.embed_model).get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    prompt_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="你是一个专业的 知识库专家，请根据实体名称列表，返回最相关的信息。"
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"实体名称列表：{entity_name_list}"
            )
        ]
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        refine_template=prompt_template,
    )
    query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, similarity_top_k=5)
    query_str = ', '.join(entity_name_list)
    return query_engine.query(query_str).response



def multi_hop_query(query: str):
    """
    执行多跳查询
    """


    # 抽取实体名称列表
    entity_name_list = company_recognize(query)

    if not entity_name_list or len(entity_name_list) == 0:
        return "未识别到实体名称"

    cypher_result = cypher_query(entity_name_list=entity_name_list)
    rag_result = rag_query(entity_name_list=entity_name_list)

    result = final_answer(question=query, recognized_result=entity_name_list, cypher_result=cypher_result, rag_result=rag_result)
    return result


def final_answer(question: str, recognized_result: list[str], cypher_result: str, rag_result: str):
    """
    生成最终答案
    """
    prompt_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""
                你是一个专业的金融分析师。请根据以下信息，以清晰、简洁的语言回答用户的问题。\n\n
                --- 实体名称列表 ---\n{entity_name_list}\n\n
                --- 知识图谱查询结果 ---\n{kg_result}\n\n
                --- 相关文档信息 ---\n{rag_context}\n\n
                """
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="用户问题：{question}"
            )
        ]
    )


    formatted_messages = prompt_template.format_messages(
        question=question,
        entity_name_list=recognized_result,
        kg_result=cypher_result,
        rag_context=rag_result,
    )

    final_response = Settings.llm.chat(messages=formatted_messages)
    return final_response.message.content.strip()