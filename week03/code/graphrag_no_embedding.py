import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM

# ============================================================================
# 配置和数据结构
# ============================================================================

# 配置日志输出，方便调试和监控
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """
    实体数据类
    
    用于存储从文本中提取的实体信息，包括：
    - name: 实体名称（如 "Paul Atreides"）
    - type: 实体类型（如 "Person", "House", "Planet"）
    - properties: 实体的额外属性（可选）
    """
    name: str                           # 实体名称，必填
    type: str                           # 实体类型，必填
    properties: Dict[str, Any] = None   # 实体属性，可选
    
    def __post_init__(self):
        """初始化后处理，确保 properties 不为 None"""
        if self.properties is None:
            self.properties = {}

@dataclass
class Relationship:
    """
    关系数据类
    
    用于存储实体间的关系信息，包括：
    - source: 源实体名称
    - target: 目标实体名称  
    - type: 关系类型（如 "PARENT_OF", "HEIR_OF"）
    - properties: 关系的额外属性（可选）
    """
    source: str                         # 源实体名称
    target: str                         # 目标实体名称
    type: str                           # 关系类型
    properties: Dict[str, Any] = None   # 关系属性，可选
    
    def __post_init__(self):
        """初始化后处理，确保 properties 不为 None"""
        if self.properties is None:
            self.properties = {}

# ============================================================================
# 核心组件类
# ============================================================================

class SimpleEntityExtractor:
    """
    功能：
    1. 使用大语言模型从文本中识别和提取实体
    2. 支持指定实体类型过滤
    3. 返回结构化的实体数据
    
    工作流程：
    文本输入 → LLM分析 → JSON解析 → 实体对象列表
    """
    
    def __init__(self, llm: OpenAILLM, node_types: List[str]):
        """
        初始化实体提取器
        
        参数：
        - llm: 大语言模型实例，用于文本分析
        - node_types: 允许提取的实体类型列表，如 ["Person", "House", "Planet"]
        """
        self.llm = llm
        self.node_types = node_types
        logger.info(f"实体提取器初始化完成，支持类型: {node_types}")
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        参数：
        - text: 待分析的文本内容
        
        返回：
        - List[Entity]: 提取到的实体列表
        
        工作步骤：
        1. 构建提示词，指定提取规则和格式
        2. 调用 LLM 进行实体识别
        3. 解析 JSON 响应
        4. 创建 Entity 对象列表
        """
        # 构建详细的提示词，指导 LLM 进行实体提取
        prompt = f"""
        你是一个专业的实体提取专家。请从以下文本中提取实体信息。

        文本内容：
        {text}

        提取规则：
        1. 只提取这些类型的实体：{', '.join(self.node_types)}
        2. 实体名称必须准确，不要修改原文中的表述
        3. 每个实体必须在文本中明确提到
        4. 如果不确定实体类型，请跳过该实体

        输出格式（必须是有效的JSON）：
        {{
            "entities": [
                {{
                    "name": "实体的完整名称",
                    "type": "实体类型（必须是指定类型之一）",
                    "properties": {{}}
                }}
            ]
        }}

        示例：
        如果文本是 "Paul Atreides is the son of Duke Leto"，输出应该是：
        {{
            "entities": [
                {{"name": "Paul Atreides", "type": "Person", "properties": {{}}}},
                {{"name": "Duke Leto", "type": "Person", "properties": {{}}}}
            ]
        }}
        """
        
        try:
            # 调用 LLM 进行实体提取
            logger.info("开始提取实体...")
            response = await self.llm.ainvoke(prompt)
            
            # 解析 JSON 响应
            result = json.loads(response.content)
            
            # 创建实体对象列表
            entities = []
            for entity_data in result.get("entities", []):
                # 验证实体类型是否在允许列表中
                if entity_data["type"] in self.node_types:
                    entity = Entity(
                        name=entity_data["name"],
                        type=entity_data["type"],
                        properties=entity_data.get("properties", {})
                    )
                    entities.append(entity)
                else:
                    logger.warning(f"跳过不支持的实体类型: {entity_data['type']}")
            
            logger.info(f"成功提取到 {len(entities)} 个实体")
            return entities
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"实体提取过程中发生错误: {e}")
            return []

class SimpleRelationshipExtractor:
    """
    功能：
    1. 识别已提取实体之间的关系
    2. 支持预定义的关系模式
    3. 验证关系的有效性
    
    工作流程：
    实体列表 + 文本 → LLM分析 → 关系验证 → 关系对象列表
    """
    
    def __init__(self, llm: OpenAILLM, relationship_types: List[str], patterns: List[Tuple[str, str, str]]):
        """
        初始化关系提取器
        
        参数：
        - llm: 大语言模型实例
        - relationship_types: 允许的关系类型列表，如 ["PARENT_OF", "HEIR_OF", "RULES"]
        - patterns: 关系模式列表，如 [("Person", "PARENT_OF", "Person")]
        """
        self.llm = llm
        self.relationship_types = relationship_types
        self.patterns = patterns
        logger.info(f"关系提取器初始化完成，支持 {len(patterns)} 种关系模式")
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        提取实体间的关系
        
        参数：
        - text: 原始文本内容
        - entities: 已提取的实体列表
        
        返回：
        - List[Relationship]: 提取到的关系列表
        
        工作步骤：
        1. 检查实体数量（至少需要2个实体才能有关系）
        2. 构建关系提取提示词
        3. 调用 LLM 识别关系
        4. 验证关系的有效性
        5. 创建 Relationship 对象列表
        """
        # 如果实体少于2个，无法形成关系
        if len(entities) < 2:
            logger.info("实体数量不足，无法提取关系")
            return []
        
        # 准备实体名称列表和关系模式描述
        entity_names = [e.name for e in entities]
        patterns_description = "\n".join([
            f"- {pattern[0]} --{pattern[1]}--> {pattern[2]}" 
            for pattern in self.patterns
        ])
        
        # 构建关系提取提示词
        prompt = f"""
        你是一个专业的关系提取专家。请从文本中识别实体间的关系。

        原始文本：
        {text}

        已识别的实体：
        {', '.join(entity_names)}

        允许的关系模式：
        {patterns_description}

        提取规则：
        1. 只提取文本中明确表达的关系
        2. 源实体和目标实体必须都在已识别实体列表中
        3. 关系类型必须符合允许的模式
        4. 关系方向很重要，注意源实体和目标实体的顺序

        输出格式（必须是有效的JSON）：
        {{
            "relationships": [
                {{
                    "source": "源实体名称",
                    "target": "目标实体名称",
                    "type": "关系类型",
                    "properties": {{}}
                }}
            ]
        }}

        示例：
        如果文本是 "Paul is the son of Duke Leto"，且实体包含 Paul Atreides 和 Duke Leto，
        输出应该是：
        {{
            "relationships": [
                {{"source": "Duke Leto", "target": "Paul Atreides", "type": "PARENT_OF", "properties": {{}}}}
            ]
        }}
        """
        
        try:
            # 调用 LLM 进行关系提取
            logger.info("开始提取关系...")
            response = await self.llm.ainvoke(prompt)
            
            # 解析 JSON 响应
            result = json.loads(response.content)
            
            # 验证和创建关系对象
            relationships = []
            for rel_data in result.get("relationships", []):
                # 验证源实体和目标实体是否存在
                source_name = rel_data.get("source", "")
                target_name = rel_data.get("target", "")
                rel_type = rel_data.get("type", "")
                
                if (source_name in entity_names and 
                    target_name in entity_names and 
                    rel_type in self.relationship_types):
                    
                    relationship = Relationship(
                        source=source_name,
                        target=target_name,
                        type=rel_type,
                        properties=rel_data.get("properties", {})
                    )
                    relationships.append(relationship)
                else:
                    logger.warning(f"跳过无效关系: {source_name} --{rel_type}--> {target_name}")
            
            logger.info(f"成功提取到 {len(relationships)} 个关系")
            return relationships
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return []
        except Exception as e:
            logger.error(f"关系提取过程中发生错误: {e}")
            return []

class SimpleGraphWriter:
    """
    功能：
    1. 将实体和关系写入 Neo4j 数据库
    2. 使用标准 Cypher 查询，不依赖 APOC 插件
    3. 支持批量操作，提高性能
    4. 自动处理重复数据（使用 MERGE）
    
    设计原则：
    - 按类型分组批量写入，减少数据库交互次数
    - 使用 MERGE 而不是 CREATE，避免重复数据
    - 完善的错误处理和日志记录
    """
    
    def __init__(self, driver):
        """
        初始化图数据写入器
        
        参数：
        - driver: Neo4j 数据库驱动实例
        """
        self.driver = driver
        logger.info("图数据写入器初始化完成")
    
    async def write_entities(self, entities: List[Entity]):
        """
        批量写入实体到 Neo4j 数据库
        
        参数：
        - entities: 要写入的实体列表
        
        工作步骤：
        1. 按实体类型分组（Person, House, Planet 等）
        2. 为每种类型批量执行写入操作
        3. 使用 MERGE 避免重复创建
        """
        if not entities:
            logger.info("没有实体需要写入")
            return
        
        # 按实体类型分组，提高批量写入效率
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # 为每种实体类型执行批量写入
        for entity_type, type_entities in entities_by_type.items():
            await self._batch_write_entities(type_entities, entity_type)
    
    async def _batch_write_entities(self, entities: List[Entity], entity_type: str):
        """
        批量写入同类型实体的内部方法
        
        参数：
        - entities: 同类型实体列表
        - entity_type: 实体类型名称
        
        使用 UNWIND 和 MERGE 实现高效的批量写入：
        - UNWIND: 将实体列表展开为单个实体
        - MERGE: 如果实体不存在则创建，存在则更新
        """
        # 构建批量写入的 Cypher 查询
        # 使用动态标签，支持不同的实体类型
        query = f"""
        UNWIND $entities AS entity
        MERGE (n:{entity_type} {{name: entity.name}})
        SET n += entity.properties
        RETURN count(n) as created
        """
        
        # 准备实体数据，转换为 Cypher 查询所需的格式
        entities_data = [
            {
                "name": entity.name,
                "properties": entity.properties
            }
            for entity in entities
        ]
        
        try:
            # 执行批量写入操作
            with self.driver.session() as session:
                result = session.run(query, entities=entities_data)
                count = result.single()["created"]
                logger.info(f"成功写入 {count} 个 {entity_type} 类型的实体")
                
        except Exception as e:
            logger.error(f"写入 {entity_type} 实体时发生错误: {e}")
    
    async def write_relationships(self, relationships: List[Relationship]):
        """
        批量写入关系到 Neo4j 数据库
        
        参数：
        - relationships: 要写入的关系列表
        
        工作步骤：
        1. 按关系类型分组
        2. 为每种关系类型批量执行写入操作
        3. 使用 MERGE 避免重复创建关系
        """
        if not relationships:
            logger.info("没有关系需要写入")
            return
        
        # 按关系类型分组
        rels_by_type = {}
        for rel in relationships:
            rel_type = rel.type
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)
        
        # 为每种关系类型执行批量写入
        for rel_type, type_rels in rels_by_type.items():
            await self._batch_write_relationships(type_rels, rel_type)
    
    async def _batch_write_relationships(self, relationships: List[Relationship], rel_type: str):
        """
        批量写入同类型关系的内部方法
        
        参数：
        - relationships: 同类型关系列表
        - rel_type: 关系类型名称
        
        查询逻辑：
        1. 通过实体名称匹配源实体和目标实体
        2. 使用 MERGE 创建关系，避免重复
        3. 设置关系属性
        """
        # 构建批量写入关系的 Cypher 查询
        query = f"""
        UNWIND $relationships AS rel
        MATCH (source {{name: rel.source}})
        MATCH (target {{name: rel.target}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += rel.properties
        RETURN count(r) as created
        """
        
        # 准备关系数据
        rels_data = [
            {
                "source": rel.source,
                "target": rel.target,
                "properties": rel.properties
            }
            for rel in relationships
        ]
        
        try:
            # 执行批量写入操作
            with self.driver.session() as session:
                result = session.run(query, relationships=rels_data)
                count = result.single()["created"]
                logger.info(f"成功写入 {count} 个 {rel_type} 类型的关系")
                
        except Exception as e:
            logger.error(f"写入 {rel_type} 关系时发生错误: {e}")

class SimpleQueryEngine:
    """
    功能：
    1. 从用户问题中提取关键词
    2. 基于关键词搜索相关实体
    3. 获取实体的图上下文信息
    4. 生成基于上下文的答案
    
    查询流程：
    用户问题 → 关键词提取 → 实体搜索 → 图遍历 → 上下文构建 → 答案生成
    """
    
    def __init__(self, driver, llm_json: OpenAILLM, llm_text: OpenAILLM):
        """
        初始化查询引擎
        
        参数：
        - driver: Neo4j 数据库驱动
        - llm_json: 用于结构化输出的 LLM（如关键词提取）
        - llm_text: 用于文本生成的 LLM（如答案生成）
        
        设计说明：
        使用两个不同配置的 LLM 实例：
        - llm_json: 配置了 json_object 响应格式，用于结构化数据提取
        - llm_text: 普通文本格式，用于自然语言生成
        """
        self.driver = driver
        self.llm_json = llm_json    # 用于结构化输出（JSON格式）
        self.llm_text = llm_text    # 用于文本生成
        logger.info("查询引擎初始化完成")
    
    async def query(self, question: str) -> str:
        """
        执行完整的 RAG 查询流程
        
        参数：
        - question: 用户提出的问题
        
        返回：
        - str: 基于知识图谱的答案
        
        查询步骤：
        1. 从问题中提取关键词
        2. 基于关键词搜索相关实体
        3. 获取实体的图上下文
        4. 基于上下文生成答案
        """
        logger.info(f"开始处理查询: {question}")
        
        # 步骤1: 从问题中提取关键词
        keywords = await self._extract_keywords(question)
        if not keywords:
            return "抱歉，无法从您的问题中提取到有效的关键词。"
        
        # 步骤2: 基于关键词搜索相关实体
        relevant_entities = await self._search_entities_by_keywords(keywords)
        if not relevant_entities:
            return "抱歉，没有找到与您问题相关的信息。"
        
        # 步骤3: 获取实体的图上下文信息
        context = await self._get_graph_context(relevant_entities)
        if not context:
            return "抱歉，无法获取相关的上下文信息。"
        
        # 步骤4: 基于上下文生成答案
        answer = await self._generate_answer(question, context)
        
        logger.info("查询处理完成")
        return answer
    
    async def _extract_keywords(self, question: str) -> List[str]:
        """
        从用户问题中提取关键词
        
        参数：
        - question: 用户问题
        
        返回：
        - List[str]: 提取到的关键词列表
        
        提取策略：
        1. 优先提取人名、地名、组织名等实体相关词汇
        2. 过滤掉停用词和无意义词汇
        3. 限制关键词数量，避免搜索范围过大
        """
        prompt = f"""
        你是一个专业的关键词提取专家。请从用户问题中提取用于搜索知识图谱的关键词。

        用户问题：
        {question}

        提取规则：
        1. 优先提取人名、地名、组织名、家族名等专有名词
        2. 提取与问题核心相关的名词
        3. 忽略疑问词（谁、什么、哪里等）和连接词
        4. 最多提取5个最重要的关键词
        5. 关键词应该是可能在知识图谱中出现的实体名称

        输出格式（必须是有效的JSON）：
        {{
            "keywords": ["关键词1", "关键词2", "关键词3"]
        }}

        示例：
        问题："谁是 Paul Atreides 的父亲？"
        输出：{{"keywords": ["Paul Atreides", "父亲"]}}
        """
        
        try:
            # 使用 JSON 格式的 LLM 进行关键词提取
            response = await self.llm_json.ainvoke(prompt)
            result = json.loads(response.content)
            keywords = result.get("keywords", [])
            
            logger.info(f"提取到关键词: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            
            # 备用方案：使用正则表达式进行简单的关键词提取
            import re
            # 提取中英文词汇，过滤掉单字符
            words = re.findall(r'\b[A-Za-z\u4e00-\u9fff]{2,}\b', question)
            fallback_keywords = words[:5]  # 最多取5个词
            
            logger.info(f"使用备用方案提取关键词: {fallback_keywords}")
            return fallback_keywords
    
    async def _search_entities_by_keywords(self, keywords: List[str]) -> List[str]:
        """
        基于关键词搜索数据库中的相关实体
        
        参数：
        - keywords: 关键词列表
        
        返回：
        - List[str]: 匹配的实体名称列表
        
        搜索策略：
        1. 使用 CONTAINS 进行模糊匹配
        2. 支持多关键词的 OR 查询
        3. 限制结果数量，避免返回过多无关实体
        """
        if not keywords:
            return []
        
        # 构建多关键词的搜索条件
        # 使用 OR 连接，只要实体名称包含任一关键词就匹配
        keyword_conditions = " OR ".join([
            f"n.name CONTAINS '{keyword}'" for keyword in keywords
        ])
        
        # 构建 Cypher 查询
        query = f"""
        MATCH (n)
        WHERE {keyword_conditions}
        RETURN DISTINCT n.name as name
        LIMIT 10
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                entity_names = [record["name"] for record in result]
                
                logger.info(f"找到 {len(entity_names)} 个相关实体: {entity_names}")
                return entity_names
                
        except Exception as e:
            logger.error(f"实体搜索失败: {e}")
            return []
    
    async def _get_graph_context(self, entity_names: List[str]) -> str:
        """
        获取实体的图上下文信息
        
        参数：
        - entity_names: 实体名称列表
        
        返回：
        - str: 格式化的图上下文信息
        
        上下文包含：
        1. 实体本身的信息（名称和类型）
        2. 实体的所有关系信息
        3. 关系连接的其他实体信息
        
        输出格式示例：
        Paul Atreides(Person) HEIR_OF House Atreides(House)
        Duke Leto(Person) PARENT_OF Paul Atreides(Person)
        """
        if not entity_names:
            return ""
        
        # 构建图遍历查询
        # 查找指定实体及其所有关系和连接的实体
        query = """
        MATCH (n)
        WHERE n.name IN $entity_names
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN n.name as entity, 
               labels(n) as entity_labels,
               type(r) as relationship, 
               connected.name as connected_entity,
               labels(connected) as connected_labels
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_names=entity_names)
                
                # 构建上下文信息
                context_parts = []
                processed_entities = set()  # 避免重复处理同一实体
                
                for record in result:
                    entity = record["entity"]
                    entity_labels = record["entity_labels"]
                    entity_type = entity_labels[0] if entity_labels else "Unknown"
                    
                    # 如果有关系信息，构建关系描述
                    if record["relationship"]:
                        rel_type = record["relationship"]
                        connected_entity = record["connected_entity"]
                        connected_labels = record["connected_labels"]
                        connected_type = connected_labels[0] if connected_labels else "Unknown"
                        
                        # 格式：实体(类型) 关系类型 连接实体(类型)
                        relation_desc = f"{entity}({entity_type}) {rel_type} {connected_entity}({connected_type})"
                        context_parts.append(relation_desc)
                    else:
                        # 如果没有关系，只记录实体本身
                        if entity not in processed_entities:
                            entity_desc = f"{entity}({entity_type})"
                            context_parts.append(entity_desc)
                            processed_entities.add(entity)
                
                context = "\n".join(context_parts)
                logger.info(f"构建图上下文，包含 {len(context_parts)} 条信息")
                return context
                
        except Exception as e:
            logger.error(f"获取图上下文失败: {e}")
            return ""
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """
        基于图上下文生成问题的答案
        
        参数：
        - question: 用户的原始问题
        - context: 从知识图谱中获取的上下文信息
        
        返回：
        - str: 生成的答案
        
        生成策略：
        1. 明确告知 LLM 基于提供的上下文回答
        2. 如果上下文信息不足，要求 LLM 明确说明
        3. 要求答案准确、简洁、有针对性
        """
        prompt = f"""
        你是一个专业的问答助手。请基于提供的知识图谱信息准确回答用户问题。

        用户问题：
        {question}

        知识图谱信息：
        {context}

        回答要求：
        1. 只基于提供的知识图谱信息进行回答
        2. 如果信息不足以回答问题，请明确说明
        3. 答案要准确、简洁、直接
        4. 如果涉及关系，请说明具体的关系类型
        5. 使用中文回答

        请直接给出答案，不需要额外的解释或格式。
        """
        
        try:
            # 使用文本生成的 LLM 生成答案
            response = await self.llm_text.ainvoke(prompt)
            answer = response.content.strip()
            
            logger.info("答案生成成功")
            return answer
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "抱歉，生成答案时出现错误，请稍后重试。"

# ============================================================================
# 主要的 GraphRAG 类
# ============================================================================

class SimpleGraphRAG:
    """
    这是整个系统的核心类，整合了所有组件：
    1. SimpleEntityExtractor - 实体提取
    2. SimpleRelationshipExtractor - 关系提取  
    3. SimpleGraphWriter - 图数据写入
    4. SimpleQueryEngine - 查询引擎
    
    主要功能：
    - build_graph(): 从文本构建知识图谱
    - query(): 基于知识图谱回答问题
    - close(): 关闭数据库连接
    
    设计特点：
    - 模块化设计，各组件职责清晰
    - 完全不依赖 APOC 插件
    - 不使用向量嵌入，降低部署复杂度
    - 完善的错误处理和日志记录
    """
    
    def __init__(self, driver, llm_json: OpenAILLM, llm_text: OpenAILLM,
                 node_types: List[str], relationship_types: List[str], 
                 patterns: List[Tuple[str, str, str]]):
        """
        初始化 GraphRAG 系统
        
        参数：
        - driver: Neo4j 数据库驱动
        - llm_json: 用于结构化输出的 LLM 实例
        - llm_text: 用于文本生成的 LLM 实例
        - node_types: 支持的实体类型列表
        - relationship_types: 支持的关系类型列表
        - patterns: 关系模式列表，定义哪些实体类型间可以有哪些关系
        """
        self.driver = driver
        self.llm_json = llm_json
        self.llm_text = llm_text
        
        # 初始化各个组件
        self.entity_extractor = SimpleEntityExtractor(llm_json, node_types)
        self.relationship_extractor = SimpleRelationshipExtractor(llm_json, relationship_types, patterns)
        self.graph_writer = SimpleGraphWriter(driver)
        self.query_engine = SimpleQueryEngine(driver, llm_json, llm_text)
        
        logger.info("SimpleGraphRAG 系统初始化完成")
        logger.info(f"支持的实体类型: {node_types}")
        logger.info(f"支持的关系类型: {relationship_types}")
        logger.info(f"关系模式数量: {len(patterns)}")
    
    async def build_graph(self, text: str):
        """
        从文本构建知识图谱
        
        参数：
        - text: 要分析的文本内容
        
        构建流程：
        1. 使用实体提取器从文本中提取实体
        2. 使用关系提取器识别实体间的关系
        3. 使用图写入器将实体和关系保存到 Neo4j
        
        这个方法是知识图谱构建的核心入口点
        """
        logger.info("=" * 50)
        logger.info("开始构建知识图谱")
        logger.info(f"输入文本长度: {len(text)} 字符")
        logger.info("=" * 50)
        
        # 步骤1: 提取实体
        logger.info("步骤1: 提取实体")
        entities = await self.entity_extractor.extract_entities(text)
        if not entities:
            logger.warning("未提取到任何实体，知识图谱构建终止")
            return
        
        # 步骤2: 提取关系
        logger.info("步骤2: 提取关系")
        relationships = await self.relationship_extractor.extract_relationships(text, entities)
        
        # 步骤3: 写入图数据
        logger.info("步骤3: 写入图数据")
        await self.graph_writer.write_entities(entities)
        await self.graph_writer.write_relationships(relationships)
        
        logger.info("=" * 50)
        logger.info("知识图谱构建完成")
        logger.info(f"共处理 {len(entities)} 个实体，{len(relationships)} 个关系")
        logger.info("=" * 50)
    
    async def query(self, question: str) -> str:
        """
        基于知识图谱回答问题
        
        参数：
        - question: 用户提出的问题
        
        返回：
        - str: 基于知识图谱的答案
        
        这个方法是问答功能的核心入口点
        """
        logger.info("=" * 50)
        logger.info("开始处理用户查询")
        logger.info(f"用户问题: {question}")
        logger.info("=" * 50)
        
        # 调用查询引擎处理问题
        answer = await self.query_engine.query(question)
        
        logger.info("=" * 50)
        logger.info("查询处理完成")
        logger.info(f"生成答案: {answer}")
        logger.info("=" * 50)
        
        return answer
    
    def close(self):
        """
        关闭数据库连接
        
        在使用完毕后调用此方法释放资源
        """
        logger.info("关闭数据库连接")
        self.driver.close()

# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    """
    演示如何使用 SimpleGraphRAG 系统
    
    这个演示展示了完整的使用流程：
    1. 连接 Neo4j 数据库
    2. 初始化 LLM 模型
    3. 创建 GraphRAG 实例
    4. 从文本构建知识图谱
    5. 进行问答查询
    """
    import nest_asyncio
    nest_asyncio.apply()  # 允许在 Jupyter 等环境中运行异步代码
    
    async def demo():
        """演示 SimpleGraphRAG 的完整功能"""
        print(" SimpleGraphRAG 演示开始")
        print("=" * 60)
        
        try:
            # 步骤1: 连接 Neo4j 数据库
            print(" 连接 Neo4j 数据库...")
            driver = GraphDatabase.driver(
                "neo4j://localhost:7687", 
                auth=("neo4j", "password")
            )
            
            # 步骤2: 初始化 LLM 模型
            print(" 初始化 LLM 模型...")
            
            # JSON 格式的 LLM，用于结构化数据提取
            llm_json = OpenAILLM(
                model_name="qwen-plus",
                model_params={
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
            )
            
            # 文本格式的 LLM，用于自然语言生成
            llm_text = OpenAILLM(
                model_name="qwen-plus",
                model_params={
                    "max_tokens": 1000,
                    "temperature": 0,
                },
            )
            
            # 步骤3: 创建 GraphRAG 实例
            print(" 创建 GraphRAG 实例...")
            graph_rag = SimpleGraphRAG(
                driver=driver,
                llm_json=llm_json,
                llm_text=llm_text,
                node_types=["Person", "House", "Planet"],           # 支持的实体类型
                relationship_types=["PARENT_OF", "HEIR_OF", "RULES"], # 支持的关系类型
                patterns=[                                          # 关系模式定义
                    ("Person", "PARENT_OF", "Person"),    # 人物之间的父子关系
                    ("Person", "HEIR_OF", "House"),       # 人物继承家族
                    ("House", "RULES", "Planet"),         # 家族统治星球
                ]
            )
            
            # 步骤4: 构建知识图谱
            print(" 构建知识图谱...")
            sample_text = """
            保罗·厄崔迪是雷托公爵和杰西卡夫人的儿子。
            保罗是厄崔迪家族的继承人，这是一个贵族家族。
            厄崔迪家族统治着卡拉丹星球。
            雷托公爵是厄崔迪家族的领袖。
            """
            
            await graph_rag.build_graph(sample_text)
            
            # 步骤5: 进行问答查询
            print(" 进行问答查询...")
            
            # 测试多个问题
            test_questions = [
                "谁是 Paul Atreides 的父亲？",
                "House Atreides 统治哪个星球？",
                "Paul Atreides 是哪个家族的继承人？"
            ]
            
            for question in test_questions:
                print(f"\n问题: {question}")
                answer = await graph_rag.query(question)
                print(f"答案: {answer}")
            
            # 步骤6: 清理资源
            print("\n 清理资源...")
            graph_rag.close()
            
            print("=" * 60)
            print(" SimpleGraphRAG 演示完成")
            
        except Exception as e:
            print(f" 演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行演示
    asyncio.run(demo())