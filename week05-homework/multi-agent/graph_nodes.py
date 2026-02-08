from graph_state import ArticleState
from agent_define import ResearchAgent, WriteAgent, ReviewAgent, PolishAgent
from langchain_core.messages import AIMessage


class GraphNodes:

    def __init__(self, mcp_tools: list):
        self.mcp_tools = {tool.name: tool for tool in mcp_tools}

    # 节点定义（research_data 需异步，因 MCP 工具仅支持 ainvoke）
    async def research_data(self, state: ArticleState):
        user_input = state.user_input
        agent = ResearchAgent(tools=list(self.mcp_tools.values()))
        response = await agent.ainvoke(user_input)
        return {
            "research_data": response,
            "messages": state.messages + [AIMessage(content=response)],
            "user_input": user_input,
        }

    def write_article(self, state: ArticleState):
        research_data_val = state.research_data
        user_input = state.user_input
        article_style = state.article_style
        article_length = state.article_length
        agent = WriteAgent()
        response = agent.invoke(
            user_input, research_data_val, article_style, article_length,
        )
        return {
            "article_draft": response,
            "messages": state.messages + [AIMessage(content=response)],
        }

    def review_article(self, state: ArticleState):
        article_draft = state.article_draft
        user_input = state.user_input
        article_style = state.article_style
        agent = ReviewAgent()
        response = agent.invoke(user_input, article_draft, article_style)
        return {
            "article_review_result": response,
            "messages": state.messages + [AIMessage(content=response)],
        }

    def polish_article(self, state: ArticleState):
        article_review_result = state.article_review_result
        article_draft = state.article_draft
        user_input = state.user_input
        article_style = state.article_style
        article_length = state.article_length
        agent = PolishAgent()
        response = agent.invoke(
            user_input, 
            article_draft, 
            article_style, 
            article_length,
            article_review_result)
        return {
            "article_polish_result": response,
            "messages": state.messages + [AIMessage(content=response)],
        }
