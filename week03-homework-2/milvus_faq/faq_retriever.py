from milvus_manager import MilvusManager
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.core import Settings
from pathlib import Path
from dotenv import load_dotenv

# 固定从 week03-homework-2 目录加载 .env，与 web 服务从哪一目录启动无关
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model=True
)
class FAQRetriever:
    def __init__(self, milvus_manager: MilvusManager):
        self.milvus_manager = milvus_manager


    def query(self, question: str):
        """
        查询 FAQ, 返回答案
        Args:
            question: 问题
        Returns:
            answer: 答案
        """


        # 日志输出匹配的 FAQ 问题和答案以及匹配度
        index = self.milvus_manager.query_collection()


        # 增加提示词模版，
        custom_refine_prompt = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="""你是一个专业的签证顾问，请根据用户的问题，返回最相关的 FAQ 答案。
                    上下文信息：{context_str}"""
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content="""用户问题：{query_str}, 请基于上下文信息回答问题。"""
                )
            ]
        )
        
        # 创建响应合成器，使用自定义提示词
        from llama_index.core.response_synthesizers import ResponseMode
        from llama_index.core.response_synthesizers import get_response_synthesizer
        
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,  # 或使用 ResponseMode.REFINE
            refine_template=custom_refine_prompt,
        )
        query_engine = index.as_query_engine(response_synthesizer=response_synthesizer, similarity_top_k=5)
        results = query_engine.query(question)

        # 获取匹配的 FAQ 问题和答案以及匹配度
        matching_results = results.source_nodes
        matching_results_list = []
        for result in matching_results:
            matching_results_list.append({
                "question": result.metadata["question"],
                "answer": result.metadata["answer"],
                "score": result.score,
                "content": result.get_content()
            })

        # 返回答案、匹配的 FAQ 问题和答案以及匹配度
        return {
            "answer": results.response,
            "matching_results": matching_results_list
        }