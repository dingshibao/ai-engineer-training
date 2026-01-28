#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from neo4j import GraphDatabase

def test_graph_traversal():
    """测试图遍历功能"""
    print("🔄 测试图遍历算法...")
    
    # 模拟数据库连接（实际使用时需要真实的Neo4j连接）
    try:
        driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
        
        # 测试连接
        with driver.session() as session:
            # 清空并创建测试数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 创建公司节点
            companies = ["A公司", "B公司", "C公司", "D公司", "E公司", "F公司", "G公司"]
            for company in companies:
                session.run("MERGE (c:Company {name: $name})", name=company)
            
            # 创建控股关系
            relationships = [
                ("A公司", "B公司"),
                ("A公司", "D公司"),
                ("B公司", "C公司"),
                ("B公司", "E公司"),
                ("C公司", "F公司"),
                ("D公司", "G公司")
            ]
            
            for parent, child in relationships:
                session.run("""
                    MATCH (p:Company {name: $parent})
                    MATCH (c:Company {name: $child})
                    MERGE (p)-[:CONTROLS]->(c)
                """, parent=parent, child=child)
            
            print("✅ 测试数据创建完成")
            
            # 测试多跳查询
            result = session.run("""
                MATCH path = (parent:Company {name: 'A公司'})-[:CONTROLS*1..]->(subsidiary:Company)
                RETURN subsidiary.name as subsidiary, 
                       length(path) as depth,
                       [node in nodes(path) | node.name] as path_nodes
                ORDER BY depth, subsidiary.name
            """)
            
            print("\n📊 A公司的子公司结构:")
            subsidiaries = []
            for record in result:
                sub_info = {
                    'subsidiary': record['subsidiary'],
                    'depth': record['depth'],
                    'path': record['path_nodes']
                }
                subsidiaries.append(sub_info)
                path_str = " → ".join(sub_info['path'])
                print(f"   第{sub_info['depth']}层: {sub_info['subsidiary']}")
                print(f"   路径: {path_str}")
            
            print(f"\n✅ 总计找到 {len(subsidiaries)} 个子公司")
            
            # 模拟问答
            print("\n🤖 模拟智能问答:")
            question = "A公司的子公司有哪些？"
            print(f"   问: {question}")
            
            if subsidiaries:
                answer_parts = ["A公司的子公司包括:"]
                for sub in subsidiaries:
                    path_str = " → ".join(sub['path'])
                    answer_parts.append(f"• {sub['subsidiary']} (路径: {path_str})")
                answer = "\n".join(answer_parts)
            else:
                answer = "A公司没有子公司"
            
            print(f"   答: {answer}")
            
        driver.close()
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 连接Neo4j失败: {e}")
        print("💡 请确保Neo4j服务正在运行，用户名密码正确")
        
        # 提供模拟结果
        print("\n🔄 使用模拟数据演示:")
        mock_subsidiaries = [
            {'subsidiary': 'B公司', 'depth': 1, 'path': ['A公司', 'B公司']},
            {'subsidiary': 'D公司', 'depth': 1, 'path': ['A公司', 'D公司']},
            {'subsidiary': 'C公司', 'depth': 2, 'path': ['A公司', 'B公司', 'C公司']},
            {'subsidiary': 'E公司', 'depth': 2, 'path': ['A公司', 'B公司', 'E公司']},
            {'subsidiary': 'G公司', 'depth': 2, 'path': ['A公司', 'D公司', 'G公司']},
            {'subsidiary': 'F公司', 'depth': 3, 'path': ['A公司', 'B公司', 'C公司', 'F公司']}
        ]
        
        print("📊 A公司的控股结构:")
        for sub in mock_subsidiaries:
            path_str = " → ".join(sub['path'])
            print(f"   第{sub['depth']}层: {sub['subsidiary']}")
            print(f"   路径: {path_str}")
        
        print(f"\n✅ 总计 {len(mock_subsidiaries)} 个子公司")
        
        print("\n🤖 智能问答演示:")
        print("   问: A公司的子公司有哪些？")
        answer_parts = ["A公司的子公司包括:"]
        for sub in mock_subsidiaries:
            path_str = " → ".join(sub['path'])
            answer_parts.append(f"• {sub['subsidiary']} (路径: {path_str})")
        answer = "\n".join(answer_parts)
        print(f"   答: {answer}")

if __name__ == "__main__":
    test_graph_traversal()