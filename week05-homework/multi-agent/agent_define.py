from pathlib import Path
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

PARENT_DIR = Path(__file__).parent.parent
load_dotenv(PARENT_DIR / ".env")

from langchain.agents import create_agent

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

class ResearchAgent:
    def __init__(self, tools: list):
        self.client = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
            你是一个研究助手，你的任务是根据用户输入，借助工具收集相关资料，生成markdown格式的结构化研究报告。
            只返回markdown格式的结构化研究报告，不要返回任何其他内容。
            研究报告的格式为：
            # 研究报告
            ## 核心概念
            ## 关键技术/方面
            ## 应用场景
            ## 未来趋势
            ## 参考文献
            """,
            debug=True,
        )

    def invoke(self, user_input: str):
        input_message = HumanMessage(content=f"用户输入：{user_input}")
        llm_response = self.client.invoke({"messages": [input_message]})
        llm_message = llm_response.get("messages")[-1].content
        return llm_message

    async def ainvoke(self, user_input: str):
        """异步调用，MCP 工具仅支持异步。"""
        input_message = HumanMessage(content=f"用户输入：{user_input}")
        llm_response = await self.client.ainvoke({"messages": [input_message]})
        llm_message = llm_response.get("messages")[-1].content
        return llm_message

class WriteAgent:
    def __init__(self):
        self.client = create_agent(
            model=llm,
            tools=[],
            system_prompt="""
            你是一个写作助手，你的任务是根据用户输入和研究数据，撰写一篇关于指定主题的文章。
            只返回文章草稿，不要返回任何其他内容。
            """,
            debug=True,
        )

    def invoke(self, user_input: str, research_data: str, article_style: str, article_length: int):
        input_message = HumanMessage(content=f"""
        请根据以下研究数据，撰写一篇关于用户输入的文章：{research_data}\n
        文章风格：{article_style}\n
        文章长度：{article_length}字左右\n
        用户输入：{user_input}\n
        """)
        llm_response = self.client.invoke({"messages": [input_message]})
        llm_message = llm_response.get("messages")[-1].content
        return llm_message

class ReviewAgent:
    def __init__(self):
        self.client = create_agent(
            model=llm,
            tools=[],
            system_prompt="""
            你是一个审核助手，你的任务是审核文章是否符合要求，并给出修改建议。
            如果文章符合要求，返回"文章符合要求"，否则返回"文章不符合要求"，并给出修改建议。
            文章要求如下：
            1. 可读性
            2. 内容质量
            3. 逻辑一致性
            """,
            debug=True,
        )

    def invoke(self, input: str, article_draft: str, article_style: str):
        input_message = HumanMessage(content=f"""
        请根据以下文章草稿，审核文章是否符合要求：{article_draft}\n
        用户输入：{input}\n
        文章风格：{article_style}\n
        """)
        llm_response = self.client.invoke({"messages": [input_message]})
        llm_message = llm_response.get("messages")[-1].content
        return llm_message

class PolishAgent:
    def __init__(self):
        self.client = create_agent(
            model=llm,
            tools=[],
            system_prompt="""
            你是一个润色助手，你的任务是根据用户输入和文章草稿，润色文章。
            只返回润色后的文章，不要返回任何其他内容。
            润色要求如下：
            1. 优化语言表格和文章结构使其更加自然
            2. 修正任何语法、拼写或标点错误
            3. 使文章更加流畅和易读
            """,
            debug=True,
        )

    def invoke(self, input: str, article_draft: str, article_style: str, article_length: int, article_review_result: str):
        
        if article_review_result.startswith("文章不符合要求"):
            input_message = HumanMessage(content=f"""
            请根据以下文章草稿，润色文章：{article_draft}\n
            用户输入：{input}\n
            文章风格：{article_style}\n
            文章长度：{article_length}字左右\n
            文章审核建议：{article_review_result}，请根据建议修改文章
            """)
        else:
            input_message = HumanMessage(content=f"""
            请根据以下文章草稿，润色文章：{article_draft}\n
            用户输入：{input}\n
            文章风格：{article_style}\n
            文章长度：{article_length}字左右\n
            """)
        llm_response = self.client.invoke({"messages": [input_message]})
        llm_message = llm_response.get("messages")[-1].content
        return llm_message