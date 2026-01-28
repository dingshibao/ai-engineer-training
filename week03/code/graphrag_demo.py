import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    name: str
    type: str

@dataclass  
class Relationship:
    source: str
    target: str
    type: str

class GraphRAG:
    """GraphRAG 核心类 - 公司控股关系演示"""
    
    def __init__(self, driver, llm_json, llm_text):
        self.driver = driver
        self.llm_json = llm_json  # 结构化输出
        self.llm_text = llm_text  # 文本生成
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """步骤1: 提取公司实体"""
        prompt = f"""
        从文本中提取公司实体：
        
        文本：{text}
        
        返回JSON格式：
        {{
            "entities": [
                {{"name": "公司名", "type": "Company"}}
            ]
        }}
        """
        
        response = await self.llm_json.ainvoke(prompt)
        result = json.loads(response.content)
        
        entities = [Entity(e["name"], e["type"]) for e in result.get("entities", [])]
        print(f" 提取到 {len(entities)} 个公司实体")
        return entities
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """步骤2: 提取控股关系"""
        entity_names = [e.name for e in entities]
        
        prompt = f"""
        从文本中提取公司间的控股关系：
        
        文本：{text}
        公司：{entity_names}
        
        返回JSON格式：
        {{
            "relationships": [
                {{"source": "母公司", "target": "子公司", "type": "CONTROLS"}}
            ]
        }}
        """
        
        response = await self.llm_json.ainvoke(prompt)
        result = json.loads(response.content)
        
        relationships = []
        for r in result.get("relationships", []):
            if r["source"] in entity_names and r["target"] in entity_names:
                relationships.append(Relationship(r["source"], r["target"], r["type"]))
        
        print(f" 提取到 {len(relationships)} 个控股关系")
        return relationships
    
    async def build_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """步骤3: 构建公司控股图谱"""
        with self.driver.session() as session:
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 写入公司实体
            for entity in entities:
                query = f"MERGE (n:{entity.type} {{name: $name}})"
                session.run(query, name=entity.name)
            
            # 写入控股关系
            for rel in relationships:
                query = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:{rel.type}]->(b)
                """
                session.run(query, source=rel.source, target=rel.target)
        
        print(f" 公司控股图谱构建完成")
    
    def find_subsidiaries_with_path(self, parent_company: str) -> List[Dict]:
        """使用图遍历算法查找所有子公司及路径"""
        with self.driver.session() as session:
            # 使用Cypher的路径查询功能实现多跳推理
            query = """
            MATCH path = (parent:Company {name: $parent_name})-[:CONTROLS*1..]->(subsidiary:Company)
            RETURN subsidiary.name as subsidiary, 
                   length(path) as depth,
                   [node in nodes(path) | node.name] as path_nodes
            ORDER BY depth, subsidiary.name
            """
            
            result = session.run(query, parent_name=parent_company)
            subsidiaries = []
            
            for record in result:
                subsidiaries.append({
                    'subsidiary': record['subsidiary'],
                    'depth': record['depth'],
                    'path': record['path_nodes']
                })
            
            return subsidiaries
    
    def visualize_control_structure(self, parent_company: str):
        """可视化控股结构"""
        subsidiaries = self.find_subsidiaries_with_path(parent_company)
        
        print(f" {parent_company} 的控股结构:")
        print("=" * 50)
        
        if not subsidiaries:
            print(f"   {parent_company} 没有子公司")
            return
        
        # 按层级分组显示
        levels = {}
        for sub in subsidiaries:
            depth = sub['depth']
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(sub)
        
        for depth in sorted(levels.keys()):
            print(f"第{depth}层子公司:")
            for sub in levels[depth]:
                path_str = " → ".join(sub['path'])
                print(f"   • {sub['subsidiary']}")
                print(f"     路径: {path_str}")
    
    async def query_graph(self, question: str) -> str:
        """步骤4: 智能问答 - 支持多跳推理"""
        # 解析问题，提取公司名
        company_name = None
        if "子公司" in question:
            # 改进的公司名提取逻辑
            import re
            # 匹配 "X公司的子公司" 模式
            match = re.search(r'([A-Z]公司)的子公司', question)
            if match:
                company_name = match.group(1)
        
        if company_name:
            # 使用图遍历查找子公司
            subsidiaries = self.find_subsidiaries_with_path(company_name)
            
            if subsidiaries:
                answer_parts = [f"{company_name}的子公司包括:"]
                for sub in subsidiaries:
                    path_str = " → ".join(sub['path'])
                    answer_parts.append(f"• {sub['subsidiary']} (路径: {path_str})")
                
                return "\n".join(answer_parts)
            else:
                return f"{company_name}没有子公司"
        
        # 处理层级问题
        if "层级" in question or "多少层" in question:
            import re
            match = re.search(r'([A-Z]公司)', question)
            if match:
                company_name = match.group(1)
                subsidiaries = self.find_subsidiaries_with_path(company_name)
                if subsidiaries:
                    max_depth = max(sub['depth'] for sub in subsidiaries)
                    return f"{company_name}共有{max_depth}层子公司，总计{len(subsidiaries)}个子公司"
                else:
                    return f"{company_name}没有子公司"
        
        # 通用图谱查询
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(m)
                RETURN n.name, type(r), m.name
                LIMIT 20
            """)
            
            context = []
            for record in result:
                context.append(f"{record[0]} {record[1]} {record[2]}")
        
        # 生成答案
        prompt = f"""
        基于公司控股图谱回答问题：
        
        问题：{question}
        
        图谱信息：
        {chr(10).join(context)}
        
        请简洁回答：
        """
        
        response = await self.llm_text.ainvoke(prompt)
        return response.content.strip()

async def demo():
    """公司控股关系 GraphRAG 演示"""
    print(" 公司控股关系 GraphRAG 演示")
    print("=" * 50)
    
    # 连接数据库
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
    
    # 初始化LLM
    llm_json = OpenAILLM(
        model_name="qwen-plus",
        model_params={"response_format": {"type": "json_object"}, "temperature": 0}
    )
    
    llm_text = OpenAILLM(
        model_name="qwen-plus", 
        model_params={"temperature": 0}
    )
    
    # 创建GraphRAG实例
    graph_rag = GraphRAG(driver, llm_json, llm_text)
    
    # 公司控股关系演示文本
    text = """
    A公司是一家大型集团公司。
    A公司控股B公司，持股比例为60%。
    A公司还控股D公司，持股比例为55%。
    B公司控股C公司，持股比例为70%。
    B公司控股E公司，持股比例为80%。
    C公司控股F公司，持股比例为65%。
    D公司控股G公司，持股比例为75%。
    """
    
    print(" 输入的公司控股信息:")
    print(text.strip())
    print()
    
    try:
        # 步骤1: 提取公司实体
        print(" 步骤1: 提取公司实体")
        entities = await graph_rag.extract_entities(text)
        for e in entities:
            print(f"   • {e.name} ({e.type})")
        print()
        
        # 步骤2: 提取控股关系  
        print(" 步骤2: 提取控股关系")
        relationships = await graph_rag.extract_relationships(text, entities)
        for r in relationships:
            print(f"   • {r.source} --{r.type}--> {r.target}")
        print()
        
        # 步骤3: 构建控股图谱
        print(" 步骤3: 构建公司控股图谱")
        await graph_rag.build_graph(entities, relationships)
        print()
        
        # 步骤4: 可视化控股结构
        print(" 步骤4: 可视化控股结构")
        graph_rag.visualize_control_structure("A公司")
        print()
        
        # 步骤5: 多跳推理问答
        print(" 步骤5: 多跳推理智能问答")
        questions = [
            "A公司的子公司有哪些？",
            "B公司的子公司有哪些？",
            "A公司有多少层级的子公司？"
        ]
        
        for question in questions:
            print(f"    问: {question}")
            answer = await graph_rag.query_graph(question)
            print(f"    答: {answer}")
            print()
        
        # 演示图遍历算法
        print(" 图遍历算法演示:")
        print("查找A公司的所有子公司及控股路径...")
        subsidiaries = graph_rag.find_subsidiaries_with_path("A公司")
        
        print(f" 多跳推理结果:")
        print(f"A公司共有 {len(subsidiaries)} 个子公司:")
        for sub in subsidiaries:
            path_str = " → ".join(sub['path'])
            print(f"   第{sub['depth']}层: {sub['subsidiary']}")
            print(f"   控股路径: {path_str}")
            print()
        
    except Exception as e:
        print(f" 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.close()
        print(" 演示完成")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(demo())