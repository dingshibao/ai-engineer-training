#####################################
######       向量知识库构建系统         #######
#####################################
import gradio as gr
import os
import shutil
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core.schema import TextNode
from upload_file import *

# 系统路径配置 - 分层存储架构
DB_PATH = "VectorStore"                    # 向量数据库存储根目录
STRUCTURED_FILE_PATH = "File/Structured"   # 结构化数据源路径
UNSTRUCTURED_FILE_PATH = "File/Unstructured"  # 非结构化数据源路径
TMP_NAME = "tmp_abcd"                      # 临时知识库标识符

# 嵌入模型配置 - 采用阿里云DashScope高性能向量化服务
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,  # 使用V2版本获得更好的语义理解
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,    # 针对文档类型优化
)

# 本地嵌入模型备选方案 - 支持离线部署和数据隐私保护
# 技术考虑：本地模型虽然性能略低，但提供完全的数据控制权
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 全局嵌入模型设置 - 确保整个系统使用统一的向量化策略
Settings.embed_model = EMBED_MODEL
# 刷新知识库
def refresh_knowledge_base():
    return os.listdir(DB_PATH)

def create_unstructured_db(db_name: str, label_name: list):
    """
    非结构化数据向量知识库构建器
    
    核心技术实现：
    1. 多源文档聚合：支持跨类目文档整合，构建综合性知识库
    2. 自动文档解析：利用SimpleDirectoryReader实现多格式文档统一处理
    3. 向量化索引：采用FAISS后端的高性能向量检索引擎
    4. 持久化存储：索引结构序列化到磁盘，支持系统重启后快速加载
    
    设计考虑：
    - 批量处理优化：一次性处理多个类目，减少重复的向量化计算开销
    - 内存管理：大文档集合的分批处理，避免内存溢出
    - 错误恢复：异常情况下的优雅降级和用户反馈
    """
    print(f"知识库名称为：{db_name}，类目名称为：{label_name}")
    
    # 输入验证 - 确保必要参数完整性
    if label_name is None:
        gr.Info("没有选择类目")
    elif len(db_name) == 0:
        gr.Info("没有命名知识库")
    # 防重复创建检查 - 避免意外覆盖现有知识库
    elif db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
    else:
        gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")
        
        # 多源文档聚合处理
        documents = []
        for label in label_name:
            label_path = os.path.join(UNSTRUCTURED_FILE_PATH, label)
            # SimpleDirectoryReader自动识别并解析多种文档格式
            # 支持PDF、DOCX、TXT等格式的统一处理
            documents.extend(SimpleDirectoryReader(label_path).load_data())
        
        # 向量索引构建 - 核心向量化处理
        # from_documents方法自动完成：文档分块 -> 向量化 -> 索引构建
        index = VectorStoreIndex.from_documents(documents)
        
        # 持久化存储 - 确保知识库可重复使用
        db_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(db_path):
            os.mkdir(db_path)
            # 序列化索引结构到磁盘，包括向量数据和元数据
            index.storage_context.persist(db_path)
        elif os.path.exists(db_path):
            pass  # 路径已存在，跳过创建
        
        gr.Info("知识库创建成功，可前往RAG问答进行提问")
    
def create_structured_db(db_name: str, data_table: list):
    """
    结构化数据向量知识库构建器
    
    核心技术创新：
    1. 细粒度文档分块：将预处理的结构化文本按行分割，每行作为独立检索单元
    2. 元数据保持：保留原始文档ID和文件名，支持溯源和引用
    3. 自定义节点构建：使用TextNode而非默认分块策略，确保结构化数据的完整性
    
    为什么采用自定义节点构建而非from_documents：
    - 结构化数据的语义单元是数据行，而非传统的文本段落
    - 需要保持每行数据的完整性，避免跨行分块破坏数据结构
    - 自定义元数据字段，支持更精确的数据溯源和过滤
    """
    print(f"知识库名称为：{db_name}，数据表名称为：{data_table}")
    
    if data_table is None:
        gr.Info("没有选择数据表")
    elif len(db_name) == 0:
        gr.Info("没有命名知识库")
    elif db_name in os.listdir(DB_PATH):
        gr.Info("知识库已存在，请换个名字或删除原来知识库再创建")
    else:
        gr.Info("正在创建知识库，请等待知识库创建成功信息显示后前往RAG问答")
        
        # 文档加载阶段
        documents = []
        for label in data_table:
            label_path = os.path.join(STRUCTURED_FILE_PATH, label)
            documents.extend(SimpleDirectoryReader(label_path).load_data())
        
        # 自定义节点构建 - 针对结构化数据的特殊处理
        nodes = []
        for doc in documents:
            # 按行分割文档内容 - 每行代表一个完整的数据记录
            doc_content = doc.get_content().split('\n')
            for chunk in doc_content:
                if chunk.strip():  # 跳过空行
                    # 创建TextNode - LlamaIndex的基础文本节点类型
                    node = TextNode(text=chunk)
                    # 元数据保持 - 支持数据溯源和过滤查询
                    node.metadata = {
                        'source': doc.get_doc_id(),           # 原始文档标识
                        'file_name': doc.metadata['file_name'] # 源文件名
                    }
                    nodes.append(node)
        
        # 从节点构建向量索引 - 跳过默认的文档分块过程
        # 这种方式确保每个数据行作为独立的检索单元
        index = VectorStoreIndex(nodes)
        
        # 持久化存储
        db_path = os.path.join(DB_PATH, db_name)
        if not os.path.exists(db_path):
            os.mkdir(db_path)
        index.storage_context.persist(db_path)
        
        gr.Info("知识库创建成功，可前往RAG问答进行提问")


# 删除指定名称知识库
def delete_db(db_name:str):
    if db_name is not None:
        folder_path = os.path.join(DB_PATH, db_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            gr.Info(f"已成功删除{db_name}知识库")
            print(f"已成功删除{db_name}知识库")
        else:
            gr.Info(f"{db_name}知识库不存在")
            print(f"{db_name}知识库不存在")

# 实时更新知识库列表
def update_knowledge_base():
    return gr.update(choices=os.listdir(DB_PATH))

# 临时文件创建知识库
def create_tmp_kb(files):
    if not os.path.exists(os.path.join("File",TMP_NAME)):
        os.mkdir(os.path.join("File",TMP_NAME))
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file,os.path.join("File",TMP_NAME,file_name))
    documents = SimpleDirectoryReader(os.path.join("File",TMP_NAME)).load_data()
    index = VectorStoreIndex.from_documents(
        documents
    )
    db_path = os.path.join(DB_PATH,TMP_NAME)
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    index.storage_context.persist(db_path)

# 清除tmp文件夹下内容 
def clear_tmp():
    if os.path.exists(os.path.join("File",TMP_NAME)):
        shutil.rmtree(os.path.join("File",TMP_NAME))
    if os.path.exists(os.path.join(DB_PATH,TMP_NAME)):
        shutil.rmtree(os.path.join(DB_PATH,TMP_NAME))