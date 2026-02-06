"""
智能客服订单查询图
支持意图识别、订单查询、多轮对话
"""
import re
from typing import TypedDict, Dict, Any, Annotated
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pathlib import Path
import os
import json
from order_tool import OrderInfo, get_order_by_id, extract_order_id
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
# 加载环境变量
parent_dir = Path(os.path.dirname(__file__)).parent
ENV_PATH = os.path.join(parent_dir, ".env")
load_dotenv(ENV_PATH)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")


class OrderState(TypedDict, total=False):
    """图状态：支持 messages 累积、query 和 order_info 传递"""
    messages: Annotated[list, add_messages]
    query: str
    response: str
    order_info: Dict[str, Any]


# ========== 节点定义 ==========

def customer_service(order_state: OrderState) -> dict:
    """
    客服服务：根据用户输入和订单查询结果（如有）提供回复
    """

    print("-"*50, "customer_service", "-"*50)
    query = order_state.get("query", "")
    order_info = order_state.get("order_info") or {}
    messages = order_state.get("messages", [])

    # 构建带上下文的 prompt（转义 JSON 中的花括号，避免被 ChatPromptTemplate 解析为变量）
    if order_info and order_info.get("order_id"):
        # 已查到订单：向用户说明订单信息
        order_summary = json.dumps(order_info, ensure_ascii=False, indent=2)
        order_summary_escaped = order_summary.replace("{", "{{").replace("}", "}}")
        system_prompt = f"""
        你是一个专业的电商客服。以下是订单查询结果，请用自然语言向用户说明：
        {order_summary_escaped}
        请用简洁友好地告知用户订单状态和关键信息。
        """
    elif order_info and order_info.get("_query_attempted"):
        # 已尝试查询但未找到：告知用户，不要再次触发 order_query
        system_prompt = """
        你是一个专业的电商客服。订单查询已完成，但未找到用户提供的订单。
        请友好地告知用户未找到该订单，建议核对订单号或联系人工客服。
        不要输出 'order_query'，直接回复用户即可。
        """
    else:
        # 未查询过：根据用户输入决定是否转接 order_query
        system_prompt = """
        你是一个专业的电商客服。请根据用户输入，提供相应的客服服务。
        - 如果用户查询的是订单相关问题且提供了订单号，请输出 'order_query'（仅此三个词）
        - 如果用户查询的是订单相关问题但没有提供订单号，请告知用户需要提供订单号
        - 其他情况请正常回复
        """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "用户输入: {query}"),
    ])

    chain = prompt_template | llm
    response = chain.invoke({"query": query})
    result_text = response.content.strip()

    return {
        "query": query,
        "response": result_text,
        "messages": messages + [HumanMessage(content=query), AIMessage(content=result_text)],
        "order_info": order_info,
    }


def order_query(order_state: OrderState) -> dict:
    """
    订单查询：直接调用 get_order_by_id 工具，不使用 agent，避免重试循环
    """
    print("-"*50, "order_query", "-"*50)
    query = order_state.get("query", "")
    messages = order_state.get("messages", [])


    order_query_agent = create_agent(
        model=llm,
        tools=[get_order_by_id, extract_order_id],
        system_prompt="""
        你是一个专业的订单查询助手。
        如果用户提供了订单号，按以下步骤进行
        1.使用 extract_order_id 工具提取订单号，如果提取到订单号，则使用 get_order_by_id 工具查询订单信息。没有提取到订单号，则直接返回 "未提取到订单号"。
        """,
        response_format=OrderInfo,
    )

    inputs = {"messages": [HumanMessage(content=query)]}
    result = order_query_agent.invoke(inputs)
    structured_response = result.get("structured_response", None)

    if structured_response:
        order_info = structured_response.model_dump()
        response_text = f"订单信息: {order_info}"
    else:
        response_text = "未找到订单"
        # 使用特殊标记，避免 customer_service 误判后再次触发 order_query 形成循环
        order_info = {"_query_attempted": True, "status": "not_found", "message": "未找到订单"}

    return {
        "query": query,
        "response": response_text,
        "messages": messages + [HumanMessage(content=query), AIMessage(content=response_text)],
        "order_info": order_info,
    }


def unknown_handler(order_state: OrderState) -> dict:
    """处理非订单相关意图"""
    print("-"*50, "unknown_handler", "-"*50)
    query = order_state.get("query", "")
    messages = order_state.get("messages", [])
    return {
        "query": query,
        "response": "对不起，目前仅支持订单查询相关服务，无法处理其他问题。",
        "messages": messages + [HumanMessage(content=query), AIMessage(content="对不起，目前仅支持订单查询相关服务，无法处理其他问题。")],
    }


# ========== 边定义 ==========

def customer_service_next_step(order_state: OrderState) -> str:
    """决定客服下一步：若需订单号则去查询，否则结束"""
    response = order_state.get("response", "")
    order_info = order_state.get("order_info") or {}
    # 若已有订单信息（含查询结果：查到或未查到），直接结束，避免循环
    if order_info and (order_info.get("_query_attempted") or order_info.get("order_id")):
        return "already_has_order_info"
    if response.strip() == "order_query":
        return "need_order_info"
    return "already_has_order_info"


def intent_recognize(order_state: OrderState) -> str:
    """识别用户意图：订单相关 vs 其他"""
    query = order_state.get("query", "")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个电商客服意图识别助手。请分析用户输入，识别其意图类型。
        - 如果用户查询的是订单相关问题（如查订单、物流、快递等），返回 'customer_service'
        - 如果用户查询的是其他问题，返回 'unknown'
        请只返回意图类型的英文标识，不要返回其他内容。
        """),
        ("human", "用户输入: {query}"),
    ])
    chain = prompt_template | llm
    response = chain.invoke({"query": query})
    return response.content.strip()


def build_order_graph():
    """构建图并编译"""

    # 支持运行时重新构建图并编译
    graph = StateGraph(OrderState)
    graph.add_node("customer_service", customer_service)
    graph.add_node("order_query", order_query)
    graph.add_node("unknown", unknown_handler)

    graph.add_conditional_edges(START, intent_recognize, {
        "customer_service": "customer_service",
        "unknown": "unknown",
    })
    graph.add_conditional_edges("customer_service", customer_service_next_step, {
        "need_order_info": "order_query",
        "already_has_order_info": END,
    })
    graph.add_edge("order_query", "customer_service")
    graph.add_edge("unknown", END)



    return graph.compile(checkpointer=memory)

