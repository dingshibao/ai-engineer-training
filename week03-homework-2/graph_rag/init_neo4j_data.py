from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


# 加载指定路径的.env文件（week03-homework-2/.env）
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# 获取环境变量（strip 避免 .env 中首尾空格导致 URI 解析失败）
NEO4J_URI = (os.getenv("NEO4J_URI") or "").strip()
NEO4J_USERNAME = (os.getenv("NEO4J_USERNAME") or "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # 密码可为空字符串，不 strip
NEO4J_DATABASE = (os.getenv("NEO4J_DATABASE") or "").strip()


def _check_neo4j_env():
    """Neo4j 连接前检查必要环境变量。"""
    if not NEO4J_URI:
        raise RuntimeError(
            "未设置 NEO4J_URI。请在 week03-homework-2/.env 中配置，例如：\n"
            "NEO4J_URI=bolt://localhost:7687\n"
            "NEO4J_USERNAME=neo4j\n"
            "NEO4J_PASSWORD=你的密码\n"
            f"当前 .env 路径: {_env_path}"
        )
    if not NEO4J_USERNAME:
        raise RuntimeError("未设置 NEO4J_USERNAME，请在 .env 中配置。")
    if NEO4J_PASSWORD is None:
        raise RuntimeError("未设置 NEO4J_PASSWORD，请在 .env 中配置。（密码可为空）")


def build_graph(file_path: Path):
    """
    从 CSV 文件读取数据并构建 Neo4j 知识图谱。
    """
    _check_neo4j_env()
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    df = pd.read_csv(file_path)

    with driver.session() as session:
        # 清空数据库以避免重复创建
        print("正在清空现有图谱数据...")
        session.run("MATCH (n) DETACH DELETE n")

        print("正在创建节点和关系...")
        # 使用 UNWIND 批量创建，效率更高
        # 1. 创建所有公司和股东节点
        query_create_nodes = """
        UNWIND $rows AS row
        MERGE (c:Entity {name: row.company_name})
        ON CREATE SET c.type = '公司'
        MERGE (s:Entity {name: row.shareholder_name})
        ON CREATE SET s.type = row.shareholder_type
        """
        session.run(query_create_nodes, rows=df.to_dict('records'))

        # 2. 创建持股关系
        query_create_rels = """
        UNWIND $rows AS row
        MATCH (shareholder:Entity {name: row.shareholder_name})
        MATCH (company:Entity {name: row.company_name})
        MERGE (shareholder)-[r:HOLDS_SHARES_IN]->(company)
        SET r.share_percentage = toFloat(row.share_percentage)
        """
        session.run(query_create_rels, rows=df.to_dict('records'))

        print("图谱节点和关系创建完成。")
        
        # 创建索引以优化查询性能
        print("正在为 'Entity' 节点的 'name' 属性创建索引...")
        try:
            session.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            print("索引创建成功。")
        except Exception as e:
            print(f"创建索引时出错: {e}")


    driver.close()
    print("图谱构建流程结束。")


if __name__ == '__main__':
    build_graph(Path(__file__).parent.parent / "data" / "shareholders.csv")