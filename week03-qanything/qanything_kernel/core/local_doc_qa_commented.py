"""
LocalDocQA 核心类详细中文注释版本
本文件展示了RAG系统的核心设计原理和实现细节
"""

from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING, SYSTEM, INSTRUCTIONS, SIMPLE_PROMPT_TEMPLATE, CUSTOM_PROMPT_TEMPLATE, \
    LOCAL_RERANK_MODEL_NAME, LOCAL_EMBED_MAX_LENGTH, SEPARATORS
from typing import List, Tuple, Union, Dict
import time
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
from scipy.stats import gmean
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.connector.rerank.rerank_for_online_client import YouDaoRerank
from qanything_kernel.connector.llm import OpenAILLM
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.core.retriever.parent_retriever import ParentRetriever
from qanything_kernel.utils.general_utils import (get_time, clear_string, get_time_async, num_tokens,
                                                  cosine_similarity, clear_string_is_equal, num_tokens_embed,
                                                  num_tokens_rerank, deduplicate_documents, replace_image_references)
from qanything_kernel.utils.custom_log import debug_logger, qa_logger, rerank_logger
from qanything_kernel.core.chains.condense_q_chain import RewriteQuestionChain
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
import copy
import requests
import json
import numpy as np
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import traceback
import re


class LocalDocQA:
    """
    本地文档问答系统核心类 - 实现完整的RAG(Retrieval-Augmented Generation)能力
    
    ==================== RAG系统核心设计原理 ====================
    
    1. 【检索阶段 (Retrieval)】
       - 向量检索: 使用embedding模型将查询转换为向量，在向量数据库中找到语义相似的文档
       - 关键词检索: 使用ElasticSearch进行精确的关键词匹配
       - 混合检索: 结合两种检索方式，提高召回率和准确性
    
    2. 【重排序阶段 (Rerank)】
       - 使用专门的重排序模型对初步检索结果进行精确排序
       - 解决向量检索可能存在的语义偏差问题
       - 提高最终检索结果的相关性
    
    3. 【上下文构建阶段】
       - 智能处理token限制，在有限的上下文窗口内最大化信息利用
       - 文档聚合和去重，避免冗余信息
       - 动态调整文档数量，确保prompt质量
    
    4. 【生成阶段 (Generation)】
       - 基于检索到的文档构建结构化prompt
       - 支持流式生成，提供实时响应体验
       - 智能处理图片引用和多媒体内容
    
    ==================== 为什么要这样设计 ====================
    
    【多模态检索的必要性】
    - 向量检索擅长语义理解，但可能错过精确匹配
    - 关键词检索擅长精确匹配，但缺乏语义理解
    - 两者结合可以显著提高检索的召回率和准确性
    
    【重排序的重要性】
    - 向量检索的相似度分数可能不够精确
    - 专门的重排序模型能更好地理解查询与文档的相关性
    - 特别是在处理复杂查询时，重排序能显著提升效果
    
    【Token优化的关键性】
    - LLM有固定的上下文窗口限制
    - 需要在有限空间内包含最相关的信息
    - 智能的token管理能提高回答质量并降低成本
    
    【流式生成的用户体验】
    - 提供实时反馈，改善用户等待体验
    - 支持长文本生成的渐进式展示
    - 便于实现打字机效果的前端展示
    """
    
    def __init__(self, port):
        """
        初始化LocalDocQA实例
        
        Args:
            port: 服务端口号
        """
        self.port = port
        
        # ========== 核心组件初始化 ==========
        self.milvus_cache = None  # Milvus向量数据库缓存
        self.embeddings: YouDaoEmbeddings = None  # 文本嵌入模型，用于将文本转换为向量
        self.rerank: YouDaoRerank = None  # 重排序模型，用于对检索结果进行精确排序
        self.chunk_conent: bool = True  # 是否启用文档分块
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD  # 向量检索分数阈值
        
        # ========== 数据存储组件 ==========
        self.milvus_kb: VectorStoreMilvusClient = None  # Milvus向量数据库客户端
        self.retriever: ParentRetriever = None  # 父级检索器，整合多种检索策略
        self.milvus_summary: KnowledgeBaseManager = None  # 知识库管理器
        self.es_client: StoreElasticSearchClient = None  # ElasticSearch客户端，用于关键词检索
        
        # ========== 工具组件 ==========
        self.session = self.create_retry_session(retries=3, backoff_factor=1)  # HTTP会话，支持重试机制
        
        # 文档分割器，用于将长文档分割成适合嵌入的小块
        self.doc_splitter = CharacterTextSplitter(
            chunk_size=LOCAL_EMBED_MAX_LENGTH / 2,  # 分块大小为最大嵌入长度的一半
            chunk_overlap=0,  # 分块间无重叠，避免信息冗余
            length_function=len  # 使用字符长度计算
        )

    @staticmethod
    def create_retry_session(retries, backoff_factor):
        """
        创建带重试机制的HTTP会话
        
        【为什么需要重试机制】
        在分布式系统中，网络请求可能因为各种原因失败：
        - 网络抖动导致的临时连接问题
        - 服务器临时过载返回5xx错误
        - DNS解析临时失败
        
        重试机制能够：
        - 提高系统的可用性和稳定性
        - 减少因临时故障导致的用户体验问题
        - 通过退避策略避免对故障服务造成更大压力
        
        Args:
            retries: 重试次数
            backoff_factor: 退避因子，控制重试间隔
            
        Returns:
            配置了重试策略的requests.Session对象
        """
        session = requests.Session()
        retry = Retry(
            total=retries,  # 总重试次数
            read=retries,   # 读取重试次数
            connect=retries,  # 连接重试次数
            backoff_factor=backoff_factor,  # 重试间隔的退避因子
            status_forcelist=[500, 502, 503, 504],  # 需要重试的HTTP状态码
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def init_cfg(self, args=None):
        """
        初始化配置 - 构建完整的RAG技术栈
        
        1. YouDaoEmbeddings (嵌入模型):
           - 将自然语言文本转换为高维向量表示
           - 支持语义相似度计算
           - 是向量检索的基础
        
        2. YouDaoRerank (重排序模型):
           - 对初步检索结果进行精确重排序
           - 比简单的向量相似度更准确
           - 特别适合处理复杂查询
        
        3. KnowledgeBaseManager (知识库管理器):
           - 管理知识库的元数据
           - 处理文档的增删改查
           - 维护文档索引和关系
        
        4. VectorStoreMilvusClient (向量数据库):
           - 高性能的向量存储和检索
           - 支持大规模向量数据
           - 提供快速的相似度搜索
        
        5. StoreElasticSearchClient (全文检索):
           - 支持关键词精确匹配
           - 提供复杂的查询语法
           - 补充向量检索的不足
        
        6. ParentRetriever (统一检索器):
           - 整合多种检索策略
           - 提供统一的检索接口
           - 支持混合检索模式
        
        Args:
            args: 可选的配置参数
        """
        self.embeddings = YouDaoEmbeddings()  # 初始化嵌入模型
        self.rerank = YouDaoRerank()  # 初始化重排序模型
        self.milvus_summary = KnowledgeBaseManager()  # 初始化知识库管理器
        self.milvus_kb = VectorStoreMilvusClient()  # 初始化向量数据库客户端
        self.es_client = StoreElasticSearchClient()  # 初始化ElasticSearch客户端
        # 初始化父级检索器，整合向量检索和关键词检索
        self.retriever = ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)

    @get_time_async
    async def get_source_documents(self, query, retriever: ParentRetriever, kb_ids, time_record, hybrid_search, top_k):
        """
        从知识库检索相关文档 - RAG的核心检索阶段
        
        【检索策略设计原理】
        
        1. 混合检索策略:
           - 向量检索: 基于语义相似度，能理解同义词和上下文
           - 关键词检索: 基于精确匹配，确保重要关键词不被遗漏
           - 两者结合: 既保证召回率又保证准确性
        
        2. 容错机制:
           - 当Milvus连接失败时自动重启客户端
           - 确保服务的高可用性
           - 避免单点故障影响整个系统
        
        3. 文档过滤:
           - 过滤已删除的文档，确保检索结果的有效性
           - 避免返回过期或无效的信息
        
        4. 分数标准化:
           - 为后续重排序提供统一的分数基准
           - 便于不同检索方式的结果融合
        
        Args:
            query: 用户查询
            retriever: 检索器实例
            kb_ids: 知识库ID列表
            time_record: 时间记录字典
            hybrid_search: 是否启用混合搜索
            top_k: 返回的文档数量
            
        Returns:
            检索到的相关文档列表
        """
        source_documents = []
        start_time = time.perf_counter()
        
        # 执行文档检索，支持向量检索和混合检索
        query_docs = await retriever.get_retrieved_documents(
            query, 
            partition_keys=kb_ids, 
            time_record=time_record,
            hybrid_search=hybrid_search, 
            top_k=top_k
        )
        
        # 容错处理：如果检索失败，重启Milvus客户端并重试
        if len(query_docs) == 0:
            debug_logger.warning("MILVUS SEARCH ERROR, RESTARTING MILVUS CLIENT!")
            retriever.vectorstore_client = VectorStoreMilvusClient()
            debug_logger.warning("MILVUS CLIENT RESTARTED!")
            query_docs = await retriever.get_retrieved_documents(
                query, 
                partition_keys=kb_ids, 
                time_record=time_record,
                hybrid_search=hybrid_search, 
                top_k=top_k
            )
        
        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")
        
        # 处理检索结果，添加元数据和分数标准化
        for idx, doc in enumerate(query_docs):
            # 过滤已删除的文档
            if retriever.mysql_client.is_deleted_file(doc.metadata['file_id']):
                debug_logger.warning(f"file_id: {doc.metadata['file_id']} is deleted")
                continue
            
            # 添加检索相关的元数据
            doc.metadata['retrieval_query'] = query  # 记录检索查询，用于后续分析
            doc.metadata['embed_version'] = self.embeddings.embed_version  # 记录嵌入模型版本
            
            # 如果没有分数，使用位置倒序作为默认分数
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 1 - (idx / len(query_docs))
            
            source_documents.append(doc)
        
        debug_logger.info(f"embed scores: {[doc.metadata['score'] for doc in source_documents]}")
        return source_documents

    def reprocess_source_documents(self, custom_llm: OpenAILLM, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> Tuple[List[Document], int, str]:
        """
        智能处理源文档以适应Token限制 - RAG系统的关键优化环节
        
        【为什么需要Token优化】
        
        1. 物理限制:
           - 每个LLM都有固定的上下文窗口大小
           - 超出限制会导致请求失败或截断
           - 需要在有限空间内最大化信息利用
        
        2. 成本考虑:
           - Token数量直接影响API调用成本
           - 减少不必要的token能显著降低费用
           - 特别是在大规模应用中成本控制很重要
        
        3. 质量保证:
           - 确保最相关的文档内容能被包含
           - 避免重要信息被截断
           - 优化prompt结构提高回答质量
        
        4. 性能优化:
           - 避免超长prompt导致的响应延迟
           - 减少网络传输时间
           - 提高整体系统响应速度
        
        【处理策略】
        - 精确计算各部分token消耗
        - 使用贪心算法优先保留高质量文档
        - 智能截断而非简单丢弃
        - 考虑文档间的关联性
        
        Args:
            custom_llm: LLM实例
            query: 用户查询
            source_docs: 源文档列表
            history: 对话历史
            prompt_template: prompt模板
            
        Returns:
            (处理后的文档列表, 可用token数量, token使用说明)
        """
        # ========== 精确计算各部分的token消耗 ==========
        
        # 查询token数(预留4倍空间，考虑编码差异)
        query_token_num = int(custom_llm.num_tokens_from_messages([query]) * 4)
        
        # 历史对话token数
        history_token_num = int(custom_llm.num_tokens_from_messages([x for sublist in history for x in sublist]))
        
        # 模板token数
        template_token_num = int(custom_llm.num_tokens_from_messages([prompt_template]))

        # 引用标签的token消耗
        reference_field_token_num = int(custom_llm.num_tokens_from_messages(
            [f"<reference>[{idx + 1}]</reference>" for idx in range(len(source_docs))]))
        
        # 计算文档可用的token数量
        # 公式: 总窗口 - 输出预留 - 安全边界 - 各固定部分
        limited_token_nums = (custom_llm.token_window - 
                             custom_llm.max_token - 
                             custom_llm.offcut_token - 
                             query_token_num - 
                             history_token_num - 
                             template_token_num - 
                             reference_field_token_num)

        # ========== 详细记录token分配情况 ==========
        debug_logger.info(f"=============================================")
        debug_logger.info(f"token_window = {custom_llm.token_window}")
        debug_logger.info(f"max_token = {custom_llm.max_token}")
        debug_logger.info(f"offcut_token = {custom_llm.offcut_token}")
        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"reference_field token nums: {reference_field_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"=============================================")

        # 生成token使用说明，用于错误提示
        tokens_msg = f"""
        token_window = {custom_llm.token_window}, max_token = {custom_llm.max_token},       
        offcut_token = {custom_llm.offcut_token}, docs_available_token_nums: {limited_token_nums}, 
        template token nums: {template_token_num}, reference_field token nums: {reference_field_token_num}, 
        query token nums: {query_token_num // 4}, history token nums: {history_token_num}
        docs_available_token_nums = token_window - max_token - offcut_token - query_token_num * 4 - history_token_num - template_token_num - reference_field_token_num
        """

        # ========== 智能文档选择策略：贪心算法装箱 ==========
        
        new_source_docs = []
        total_token_num = 0
        not_repeated_file_ids = []  # 避免重复计算同一文件的headers

        for doc in source_docs:
            headers_token_num = 0
            file_id = doc.metadata['file_id']
            
            # 只为每个文件计算一次headers的token消耗
            if file_id not in not_repeated_file_ids:
                not_repeated_file_ids.append(file_id)
                if 'headers' in doc.metadata:
                    headers = f"headers={doc.metadata['headers']}"
                    headers_token_num = custom_llm.num_tokens_from_messages([headers])
            
            # 移除图片引用，只计算文本内容的token
            doc_valid_content = re.sub(r'!\[figure\]\(.*?\)', '', doc.page_content)
            doc_token_num = custom_llm.num_tokens_from_messages([doc_valid_content])
            doc_token_num += headers_token_num
            
            # 贪心策略：如果当前文档能放入，就添加；否则停止
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                break  # token预算用完，停止添加文档

        debug_logger.info(f"new_source_docs token nums: {custom_llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs, limited_token_nums, tokens_msg

    def generate_prompt(self, query, source_docs, prompt_template):
        """
        生成最终的LLM输入prompt - RAG系统的上下文构建阶段
        
        【Prompt构建的设计原理】
        
        1. 结构化引用:
           - 使用<reference>标签清晰标识每个文档来源
           - 便于LLM理解文档边界和来源
           - 支持用户追溯答案来源
        
        2. 内容优化:
           - 移除图片引用，专注于文本内容
           - 避免无关信息干扰LLM理解
           - 保持prompt的简洁性
        
        3. 文件去重:
           - 同一文件的多个片段合并处理
           - 避免重复的headers信息
           - 减少token浪费
        
        4. 编号索引:
           - 为每个文档分配唯一编号
           - 便于LLM在回答中引用具体文档
           - 支持用户验证答案来源
        
        【Prompt模板设计】
        - {{context}}: 插入检索到的文档内容
        - {{question}}: 插入用户查询
        - 模板支持自定义，适应不同场景需求
        
        Args:
            query: 用户查询
            source_docs: 检索到的相关文档
            prompt_template: prompt模板
            
        Returns:
            构建好的完整prompt
        """
        if source_docs:
            context = ''
            not_repeated_file_ids = []
            
            for doc in source_docs:
                # 移除图片引用，只保留文本内容用于LLM理解
                doc_valid_content = re.sub(r'!\[figure\]\(.*?\)', '', doc.page_content)
                file_id = doc.metadata['file_id']
                
                # 处理新文件：添加reference标签和headers
                if file_id not in not_repeated_file_ids:
                    # 关闭上一个reference标签
                    if len(not_repeated_file_ids) != 0:
                        context += '</reference>\n'
                    
                    not_repeated_file_ids.append(file_id)
                    
                    # 添加文档headers信息（如果存在）
                    if 'headers' in doc.metadata:
                        headers = f"headers={doc.metadata['headers']}"
                        context += f"<reference {headers}>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                    else:
                        context += f"<reference>[{len(not_repeated_file_ids)}]" + '\n' + doc_valid_content + '\n'
                else:
                    # 同一文件的后续片段，直接添加内容
                    context += doc_valid_content + '\n'
            
            # 关闭最后一个reference标签
            context += '</reference>\n'

            # 将上下文和查询插入到prompt模板中
            prompt = prompt_template.replace("{{context}}", context).replace("{{question}}", query)
        else:
            # 没有检索到文档时，只插入查询
            prompt = prompt_template.replace("{{question}}", query)
        
        return prompt

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, top_k, web_chunk_size,
                                         chat_history=None, streaming: bool = STREAMING, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        """
        获取基于知识库的答案 - RAG系统的主流程
        
        【完整的RAG流程】
        
        1. 查询预处理:
           - 如果有对话历史，使用RewriteQuestionChain重写查询
           - 提取关键信息，优化检索效果
        
        2. 文档检索:
           - 从知识库检索相关文档
           - 可选择启用网络搜索补充
           - 支持混合检索模式
        
        3. 结果重排序:
           - 使用重排序模型优化检索结果
           - 过滤低分文档，提高质量
           - 动态调整文档数量
        
        4. 特殊处理:
           - FAQ完全匹配检测
           - 高分FAQ文档优先处理
           - 支持多种prompt模板
        
        5. 答案生成:
           - 构建优化的prompt
           - 流式生成答案
           - 处理图片和多媒体内容
        
        【为什么需要这样的复杂流程】
        - 每个步骤都针对特定问题进行优化
        - 多层过滤确保最终结果的质量
        - 支持多种使用场景和需求
        - 提供完整的可观测性和调试信息
        
        Args:
            model: 使用的LLM模型
            max_token: 最大token数
            kb_ids: 知识库ID列表
            query: 用户查询
            retriever: 检索器
            custom_prompt: 自定义prompt
            time_record: 时间记录
            temperature: 生成温度
            api_base: API基础URL
            api_key: API密钥
            api_context_length: API上下文长度
            top_p: nucleus sampling参数
            top_k: 检索文档数量
            web_chunk_size: 网络搜索分块大小
            chat_history: 对话历史
            streaming: 是否流式输出
            rerank: 是否启用重排序
            only_need_search_results: 是否只需要搜索结果
            need_web_search: 是否需要网络搜索
            hybrid_search: 是否启用混合搜索
            
        Yields:
            (response, history): 响应和更新后的历史
        """
        # 初始化LLM客户端
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)
        
        if chat_history is None:
            chat_history = []
        
        retrieval_query = query
        condense_question = query
        
        # ========== 查询重写阶段 ==========
        if chat_history:
            # 格式化对话历史
            formatted_chat_history = []
            for msg in chat_history:
                formatted_chat_history += [
                    HumanMessage(content=msg[0]),
                    AIMessage(content=msg[1]),
                ]
            debug_logger.info(f"formatted_chat_history: {formatted_chat_history}")

            # 使用查询重写链优化查询！！！
            rewrite_q_chain = RewriteQuestionChain(model_name=model, openai_api_base=api_base, openai_api_key=api_key)
            full_prompt = rewrite_q_chain.condense_q_prompt.format(
                chat_history=formatted_chat_history,
                question=query
            )
            
            # 确保prompt不超过token限制
            while custom_llm.num_tokens_from_messages([full_prompt]) >= 4096 - 256:
                formatted_chat_history = formatted_chat_history[2:]
                full_prompt = rewrite_q_chain.condense_q_prompt.format(
                    chat_history=formatted_chat_history,
                    question=query
                )
            
            debug_logger.info(f"Subtract formatted_chat_history: {len(chat_history) * 2} -> {len(formatted_chat_history)}")
            
            try:
                t1 = time.perf_counter()
                condense_question = await rewrite_q_chain.condense_q_chain.ainvoke({
                    "chat_history": formatted_chat_history,
                    "question": query,
                })
                t2 = time.perf_counter()
                time_record['condense_q_chain'] = round(t2 - t1, 2)
                time_record['rewrite_completion_tokens'] = custom_llm.num_tokens_from_messages([condense_question])
                debug_logger.info(f"condense_q_chain time: {time_record['condense_q_chain']}s")
            except Exception as e:
                debug_logger.error(f"condense_q_chain error: {e}")
                condense_question = query
            
            debug_logger.info(f"condense_question: {condense_question}")
            time_record['rewrite_prompt_tokens'] = custom_llm.num_tokens_from_messages([full_prompt, condense_question])
            
            # 判断重写后的查询是否有显著变化
            if clear_string(condense_question) != clear_string(query):
                retrieval_query = condense_question

        # ========== 文档检索阶段 ==========
        if kb_ids:
            source_documents = await self.get_source_documents(retrieval_query, retriever, kb_ids, time_record,
                                                               hybrid_search, top_k)
        else:
            source_documents = []

        # ========== 网络搜索补充 ==========
        if need_web_search:
            t1 = time.perf_counter()
            web_search_results = self.web_page_search(query, top_k=3)
            
            # 对网络搜索结果进行分块处理
            web_splitter = RecursiveCharacterTextSplitter(
                separators=SEPARATORS,
                chunk_size=web_chunk_size,
                chunk_overlap=int(web_chunk_size / 4),
                length_function=num_tokens_embed,
            )
            web_search_results = web_splitter.split_documents(web_search_results)

            # 为网络搜索结果分配doc_id
            current_doc_id = 0
            current_file_id = web_search_results[0].metadata['file_id']
            for doc in web_search_results:
                if doc.metadata['file_id'] == current_file_id:
                    doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
                    current_doc_id += 1
                else:
                    current_file_id = doc.metadata['file_id']
                    current_doc_id = 0
                    doc.metadata['doc_id'] = current_file_id + '_' + str(current_doc_id)
                    current_doc_id += 1
                
                # 将文档添加到知识库管理器
                doc_json = doc.to_json()
                if doc_json['kwargs'].get('metadata') is None:
                    doc_json['kwargs']['metadata'] = doc.metadata
                self.milvus_summary.add_document(doc_id=doc.metadata['doc_id'], json_data=doc_json)

            t2 = time.perf_counter()
            time_record['web_search'] = round(t2 - t1, 2)
            source_documents += web_search_results

        # ========== 文档去重 ==========
        source_documents = deduplicate_documents(source_documents)
        
        # ========== 重排序阶段 ==========
        if rerank and len(source_documents) > 1 and num_tokens_rerank(query) <= 300:
            try:
                t1 = time.perf_counter()
                debug_logger.info(f"use rerank, rerank docs num: {len(source_documents)}")
                source_documents = await self.rerank.arerank_documents(condense_question, source_documents)
                t2 = time.perf_counter()
                time_record['rerank'] = round(t2 - t1, 2)
                
                # ========== 两层过滤策略：确保文档质量的双重保障 ==========
                """
                为什么需要两层过滤？
                
                【问题背景】
                重排序模型虽然比向量相似度更准确，但仍可能出现：
                1. 分数分布不均：有些查询所有文档分数都很低，有些都很高
                2. 质量断层：前几个文档质量很好，后面突然下降
                3. 噪声文档：个别不相关文档混入高分区间
                
                【解决方案：两层过滤】
                第一层：绝对阈值过滤 - 确保基本质量标准
                第二层：相对差异过滤 - 确保质量一致性
                """
                debug_logger.info(f"rerank step1 num: {len(source_documents)}")
                debug_logger.info(f"rerank step1 scores: {[doc.metadata['score'] for doc in source_documents]}")
                
                if len(source_documents) > 1:
                    # ========== 第一层过滤：绝对分数阈值 ==========
                    """
                    目的：过滤掉明显不相关的文档
                    阈值：0.28 (经验值，基于大量测试数据)
                    
                    原理：
                    - 相关文档分数通常 > 0.4
                    - 不相关文档分数通常 < 0.3  
                    - 0.28是一个保守的分界线，确保基本质量
                    
                    示例：
                    原始分数：[0.85, 0.72, 0.45, 0.23, 0.15]
                    过滤后：  [0.85, 0.72, 0.45] (0.23和0.15被过滤)
                    """
                    if filtered_documents := [doc for doc in source_documents if doc.metadata['score'] >= 0.28]:
                        source_documents = filtered_documents
                    
                    debug_logger.info(f"rerank step2 num: {len(source_documents)}")
                    
                    # ========== 第二层过滤：相对分数差异 ==========
                    """
                    目的：确保文档间质量的连续性，避免质量断层
                    阈值：50% 相对差异
                    
                    原理：
                    - 计算每个文档与最高分文档的相对差异
                    - 相对差异 = (最高分 - 当前分) / 最高分
                    - 如果差异 > 50%，说明质量下降明显，停止添加
                    
                    示例：
                    文档A: 0.85 (基准，必保留)
                    文档B: 0.72 → 差异 = (0.85-0.72)/0.85 = 15.3% ✓ 保留
                    文档C: 0.45 → 差异 = (0.85-0.45)/0.85 = 47.1% ✓ 保留  
                    文档D: 0.35 → 差异 = (0.85-0.35)/0.85 = 58.8% ✗ 停止
                    

                    为什么用相对差异而不是绝对差异？
                    - 适应不同的分数分布范围
                    - 在高分区间(0.8-1.0)和低分区间(0.3-0.5)都能有效工作
                    - 更符合人类对质量差异的感知
                    """
                    saved_docs = [source_documents[0]]  # 最高分文档必须保留
                    for doc in source_documents[1:]:
                        debug_logger.info(f"rerank doc score: {doc.metadata['score']}")
                        # 计算与最高分文档的相对差异
                        relative_difference = (saved_docs[0].metadata['score'] - doc.metadata['score']) / saved_docs[0].metadata['score']
                        if relative_difference > 0.5:  # 质量下降超过50%，停止添加
                            debug_logger.info(f"Quality drop too much: {relative_difference:.1%}, stopping at score {doc.metadata['score']}")
                            break
                        else:
                            saved_docs.append(doc)
                    source_documents = saved_docs
                    debug_logger.info(f"rerank step3 num: {len(source_documents)}")
            except Exception as e:
                time_record['rerank'] = 0.0
                debug_logger.error(f"query {query}: kb_ids: {kb_ids}, rerank error: {traceback.format_exc()}")

        # 限制最终文档数量
        source_documents = source_documents[:top_k]

        # 清理文档内容，移除headers标记
        for doc in source_documents:
            doc.page_content = re.sub(r'^\[headers\]\(.*?\)\n', '', doc.page_content)

        # ========== FAQ特殊处理 ==========
        # 优先处理高分FAQ文档
        high_score_faq_documents = [doc for doc in source_documents if
                                    doc.metadata['file_name'].endswith('.faq') and doc.metadata['score'] >= 0.9]
        if high_score_faq_documents:
            source_documents = high_score_faq_documents
        
        # FAQ完全匹配检测
        for doc in source_documents:
            if doc.metadata['file_name'].endswith('.faq') and clear_string_is_equal(
                    doc.metadata['faq_dict']['question'], query):
                debug_logger.info(f"match faq question: {query}")
                if only_need_search_results:
                    yield source_documents, None
                    return
                res = doc.metadata['faq_dict']['answer']
                async for response, history in self.generate_response(query, res, condense_question, source_documents,
                                                                      time_record, chat_history, streaming, 'MATCH_FAQ'):
                    yield response, history
                return

        # ========== Prompt构建阶段 ==========
        today = time.strftime("%Y-%m-%d", time.localtime())
        now = time.strftime("%H:%M:%S", time.localtime())

        extra_msg = None
        total_images_number = 0
        retrieval_documents = []
        
        if source_documents:
            # 选择prompt模板
            if custom_prompt:
                prompt_template = CUSTOM_PROMPT_TEMPLATE.replace("{{custom_prompt}}", custom_prompt)
            else:
                system_prompt = SYSTEM.replace("{{today_date}}", today).replace("{{current_time}}", now)
                prompt_template = PROMPT_TEMPLATE.replace("{{system}}", system_prompt).replace("{{instructions}}", INSTRUCTIONS)

            t1 = time.perf_counter()
            retrieval_documents, limited_token_nums, tokens_msg = self.reprocess_source_documents(
                custom_llm=custom_llm,
                query=query,
                source_docs=source_documents,
                history=chat_history,
                prompt_template=prompt_template
            )

            # 检查token限制导致的文档裁切
            if len(retrieval_documents) < len(source_documents):
                if len(retrieval_documents) == 0:
                    debug_logger.error(f"limited_token_nums: {limited_token_nums} < {web_chunk_size}!")
                    res = (
                        f"抱歉，由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums} < 文本分片大小: {web_chunk_size})，"
                        f"\n无法保证回答质量，请在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。"
                        f"\n计算方式：{tokens_msg}")
                    async for response, history in self.generate_response(query, res, condense_question, source_documents,
                                                                          time_record, chat_history, streaming,
                                                                          'TOKENS_NOT_ENOUGH'):
                        yield response, history
                    return

                extra_msg = (
                    f"\n\nWARNING: 由于留给相关文档使用的token数量不足(docs_available_token_nums: {limited_token_nums})，"
                    f"\n检索到的部分文档chunk被裁切，原始来源数量：{len(source_documents)}，裁切后数量：{len(retrieval_documents)}，"
                    f"\n可能会影响回答质量，尤其是问题涉及的相关内容较多时。"
                    f"\n可在模型配置中提高【总Token数量】或减少【输出Tokens数量】或减少【上下文消息数量】再继续提问。\n")

            # 进一步处理文档
            source_documents, retrieval_documents = await self.prepare_source_documents(
                custom_llm, retrieval_documents, limited_token_nums, rerank)

            # 处理图片引用
            for doc in source_documents:
                if doc.metadata.get('images', []):
                    total_images_number += len(doc.metadata['images'])
                    doc.page_content = replace_image_references(doc.page_content, doc.metadata['file_id'])
            
            debug_logger.info(f"total_images_number: {total_images_number}")
            t2 = time.perf_counter()
            time_record['reprocess'] = round(t2 - t1, 2)
        else:
            # 没有文档时的prompt模板
            if custom_prompt:
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", custom_prompt)
            else:
                simple_custom_prompt = """
                - If you cannot answer based on the given information, you will return the sentence "抱歉，已知的信息不足，因此无法回答。". 
                """
                prompt_template = SIMPLE_PROMPT_TEMPLATE.replace("{{today}}", today).replace("{{now}}", now).replace(
                    "{{custom_prompt}}", simple_custom_prompt)

        # 如果只需要搜索结果，直接返回
        if only_need_search_results:
            yield source_documents, None
            return

        # ========== 答案生成阶段 ==========
        t1 = time.perf_counter()
        has_first_return = False
        acc_resp = ''
        
        # 生成最终prompt
        prompt = self.generate_prompt(query=query, source_docs=source_documents, prompt_template=prompt_template)
        est_prompt_tokens = num_tokens(prompt) + num_tokens(str(chat_history))
        
        # 流式生成答案
        async for answer_result in custom_llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
            resp = answer_result.llm_output["answer"]
            if 'answer' in resp:
                acc_resp += json.loads(resp[6:])['answer']
            
            prompt = answer_result.prompt
            history = answer_result.history
            total_tokens = answer_result.total_tokens
            prompt_tokens = answer_result.prompt_tokens
            completion_tokens = answer_result.completion_tokens
            history[-1][0] = query
            
            response = {
                "query": query,
                "prompt": prompt,
                "result": resp,
                "condense_question": condense_question,
                "retrieval_documents": retrieval_documents,
                "source_documents": source_documents
            }
            
            # 记录token使用情况
            time_record['prompt_tokens'] = prompt_tokens if prompt_tokens != 0 else est_prompt_tokens
            time_record['completion_tokens'] = completion_tokens if completion_tokens != 0 else num_tokens(acc_resp)
            time_record['total_tokens'] = total_tokens if total_tokens != 0 else time_record['prompt_tokens'] + time_record['completion_tokens']
            
            # 记录首次返回时间
            if has_first_return is False:
                first_return_time = time.perf_counter()
                has_first_return = True
                time_record['llm_first_return'] = round(first_return_time - t1, 2)
            
            # 处理生成完成
            if resp[6:].startswith("[DONE]"):
                # 添加额外消息（如果有）
                if extra_msg is not None:
                    msg_response = {
                        "query": query,
                        "prompt": prompt,
                        "result": f"data: {json.dumps({'answer': extra_msg}, ensure_ascii=False)}",
                        "condense_question": condense_question,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents
                    }
                    yield msg_response, history
                
                last_return_time = time.perf_counter()
                time_record['llm_completed'] = round(last_return_time - t1, 2) - time_record['llm_first_return']
                history[-1][1] = acc_resp
                
                # 处理图片展示
                if total_images_number != 0:
                    docs_with_images = [doc for doc in source_documents if doc.metadata.get('images', [])]
                    time1 = time.perf_counter()
                    relevant_docs = await self.calculate_relevance_optimized(
                        question=query,
                        llm_answer=acc_resp,
                        reference_docs=docs_with_images,
                        top_k=1
                    )
                    show_images = ["\n### 引用图文如下：\n"]
                    for doc in relevant_docs:
                        for image in doc['document'].metadata.get('images', []):
                            image_str = replace_image_references(image, doc['document'].metadata['file_id'])
                            debug_logger.info(f"image_str: {image} -> {image_str}")
                            show_images.append(image_str + '\n')
                    
                    debug_logger.info(f"show_images: {show_images}")
                    time_record['obtain_images'] = round(time.perf_counter() - last_return_time, 2)
                    time2 = time.perf_counter()
                    debug_logger.info(f"obtain_images time: {time2 - time1}s")
                    time_record["obtain_images_time"] = round(time2 - time1, 2)
                    if len(show_images) > 1:
                        response['show_images'] = show_images
            
            yield response, history

    @staticmethod
    async def generate_response(query, res, condense_question, source_documents, time_record, chat_history, streaming, prompt):
        """
        生成标准化的响应格式
        
        【为什么需要标准化响应】
        - 确保所有响应都包含必要的元数据
        - 支持流式和非流式两种模式
        - 便于前端统一处理
        - 提供完整的调试信息
        
        Args:
            query: 原始查询
            res: 生成的答案
            condense_question: 重写后的查询
            source_documents: 源文档
            time_record: 时间记录
            chat_history: 对话历史
            streaming: 是否流式输出
            prompt: prompt类型标识
            
        Yields:
            (response, history): 响应和更新后的历史
        """
        history = chat_history + [[query, res]]

        if streaming:
            res = 'data: ' + json.dumps({'answer': res}, ensure_ascii=False)

        response = {
            "query": query,
            "prompt": prompt,
            "result": res,
            "condense_question": condense_question,
            "retrieval_documents": source_documents,
            "source_documents": source_documents
        }

        # 确保时间记录的完整性
        if 'llm_completed' not in time_record:
            time_record['llm_completed'] = 0.0
        if 'total_tokens' not in time_record:
            time_record['total_tokens'] = 0
        if 'prompt_tokens' not in time_record:
            time_record['prompt_tokens'] = 0
        if 'completion_tokens' not in time_record:
            time_record['completion_tokens'] = 0

        yield response, history

        # 流式输出结束标志
        if streaming:
            response['result'] = "data: [DONE]\n\n"
            yield response, history