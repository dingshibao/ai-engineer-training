# RAG对话系统核心引擎 - 集成检索增强生成技术
import os
from openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

# 重排序模块导入 - 采用优雅降级策略
try:
    from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
except ImportError:
    print("Warning: DashScopeRerank not found, will skip reranking")
    DashScopeRerank = None
    # 设计理念：系统在缺少重排序组件时仍能正常工作
    # 通过None值标记和条件检查实现功能的优雅降级

from create_kb import *

# 系统配置常量
DB_PATH = "VectorStore"  # 向量数据库根路径
TMP_NAME = "tmp_abcd"    # 临时知识库标识符

# 嵌入模型配置 - 与知识库构建保持一致
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

# 本地嵌入模型备选方案
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 全局嵌入模型设置 - 确保检索和构建使用相同的向量空间
Settings.embed_model = EMBED_MODEL

def get_model_response(multi_modal_input, history, model, temperature, max_tokens, history_round, db_name, similarity_threshold, chunk_cnt):
    """
    RAG对话系统核心响应生成器
    
    技术架构设计：
    1. 多模态输入处理：支持文本+文件混合输入模式
    2. 动态知识库切换：临时文件优先级高于预设知识库
    3. 两阶段检索策略：粗排（向量相似度）+ 精排（重排序模型）
    4. 流式响应生成：实时输出提升用户体验
    5. 异常容错机制：检索失败时降级为纯LLM对话
    
    核心算法流程：
    输入解析 -> 知识库选择 -> 向量检索 -> 重排序 -> 上下文构建 -> LLM生成
    """
    # 提取用户查询 - 从对话历史获取最新用户输入
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']
    
    # 动态知识库选择策略
    # 优先级：临时上传文件 > 用户选择的知识库
    if os.path.exists(os.path.join("File", TMP_NAME)):
        db_name = TMP_NAME  # 使用临时知识库
    else:
        if tmp_files:
            # 实时构建临时知识库 - 支持即时文档问答
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME
    
    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    
    try:
        # 重排序器初始化 - 采用条件性实例化避免导入错误
        if DashScopeRerank is not None:
            dashscope_rerank = DashScopeRerank(
                top_n=chunk_cnt,           # 重排序后保留的文档数量
                return_documents=True      # 返回完整文档而非仅ID
            )
        else:
            dashscope_rerank = None
        
        # 向量索引加载 - 从持久化存储恢复索引结构
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH, db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")
        
        # 检索器配置 - 第一阶段粗排检索
        retriever_engine = index.as_retriever(
            similarity_top_k=20,  # 粗排阶段召回更多候选文档
        )
        
        # 向量相似度检索 - 基于查询向量的语义匹配
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")
        
        # 第二阶段精排处理 - 使用专门的重排序模型
        try:
            if dashscope_rerank is not None:
                # 重排序的技术优势：
                # 1. 考虑查询与文档的深层语义关系
                # 2. 基于Transformer的交互式编码
                # 3. 相比纯向量相似度有更高的准确性
                results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
                print(f"rerank成功，重排后的chunk为：{results}")
            else:
                # 降级策略：直接使用向量相似度排序结果
                results = retrieve_chunk[:chunk_cnt]
                print(f"未使用rerank，chunk为：{results}")
        except Exception as rerank_error:
            # 重排序失败的容错处理
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")
        
        # 上下文文本构建 - 基于相似度阈值过滤
        chunk_text = ""      # 用于LLM输入的上下文
        chunk_show = ""      # 用于用户界面显示的召回文本
        
        for i in range(len(results)):
            # 相似度阈值过滤 - 排除低相关性文档减少噪声
            if results[i].score >= similarity_threshold:
                chunk_text += f"## {i+1}:\n {results[i].text}\n"
                chunk_show += f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score, 2)}\n"
        
        print(f"已获取chunk：{chunk_text}")
        
        # RAG提示词模板构建 - 结合检索内容和用户查询
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
        
    except Exception as e:
        # 检索系统异常时的降级策略 - 退化为纯LLM对话
        print(f"异常信息：{e}")
        prompt_template = prompt  # 直接使用原始查询
        chunk_show = ""
    
    # 对话历史初始化 - 为流式响应预留位置
    history[-1][-1] = ""
    
    # OpenAI兼容客户端初始化 - 支持DashScope API
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 对话上下文构建 - 实现多轮对话记忆
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []
    
    # 上下文窗口管理 - 控制token消耗和对话连贯性
    history_round = min(len(history), history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    
    # 当前查询添加到消息列表
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages
    
    # 流式响应生成 - 实时输出提升用户体验
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 控制生成的随机性
        max_tokens=max_tokens,    # 限制响应长度
        stream=True               # 启用流式输出
    )
    
    # 流式响应处理 - 逐步构建完整回答
    assistant_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            assistant_response += chunk.choices[0].delta.content
            history[-1][-1] = assistant_response
            # 生成器模式 - 实时返回更新的对话历史和召回文本
            yield history, chunk_show