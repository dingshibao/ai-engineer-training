"""电商订单处理工作流图定义

使用LangGraph构建订单处理流程，包含意图识别、大模型处理和条件路由
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
import json


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    user_id: str
    session_id: str


@dataclass
class State:
    """电商订单处理状态

    定义工作流的状态结构
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    user_input: str = ""
    intent: str = ""
    order_info: Dict[str, Any] = None
    response: str = ""
    next_action: str = ""
    messages: list = None

    def __post_init__(self):
        if self.order_info is None:
            self.order_info = {}
        if self.messages is None:
            self.messages = []


def intent_recognition_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    意图识别节点
    使用通义大模型识别用户输入的订单相关意图
    """
    from langchain_community.llms import Tongyi
    import os
    
    user_input = state.user_input
    
    # 初始化通义千问模型
    try:
        llm = Tongyi(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            model_name="qwen-turbo"
        )
        
        # 构建意图识别提示
        intent_prompt = f"""
            你是一个专业的电商客服意图识别助手。请分析用户输入，识别其意图类型。

            用户输入: {user_input}

            请从以下意图类型中选择最匹配的一个：
            1. query_order - 查询订单（用户想查询订单状态、物流信息等）
            2. modify_order - 修改订单（用户想修改订单信息、收货地址等）
            3. cancel_order - 取消订单（用户想取消或退订订单）
            4. customer_service - 客服咨询（用户需要人工客服帮助、投诉等）
            5. payment_issue - 支付问题（用户遇到支付相关问题）
            6. product_inquiry - 商品咨询（用户咨询商品信息）
            7. unknown - 未知意图（无法归类到以上类型）

            请只返回意图类型的英文标识，不要返回其他内容。
            """
        
        # 调用通义千问进行意图识别
        intent_result = llm.invoke(intent_prompt).strip()
        
        # 验证返回的意图是否有效
        valid_intents = ["query_order", "modify_order", "cancel_order", "customer_service", 
                        "payment_issue", "product_inquiry", "unknown"]
        
        if intent_result not in valid_intents:
            intent = "unknown"
        else:
            intent = intent_result
            
    except Exception as e:
        print(f"通义千问意图识别失败: {e}")
        # 降级到关键词匹配
        intent = "unknown"
        if any(keyword in user_input.lower() for keyword in ["查询", "查看", "订单状态"]):
            intent = "query_order"
        elif any(keyword in user_input.lower() for keyword in ["修改", "更改", "变更"]):
            intent = "modify_order"
        elif any(keyword in user_input.lower() for keyword in ["取消", "退单", "撤销"]):
            intent = "cancel_order"
        elif any(keyword in user_input.lower() for keyword in ["投诉", "问题", "客服"]):
            intent = "customer_service"
    
    print(f"意图识别结果: {intent}")
    
    return {
        "intent": intent,
        "messages": state.messages + [{"role": "system", "content": f"识别到意图: {intent}"}]
    }


def tongyi_llm_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    通义大模型处理节点
    调用通义千问处理订单相关问题
    """
    from langchain_community.llms import Tongyi
    import os
    
    user_input = state.user_input
    intent = state.intent
    
    try:
        # 初始化通义千问模型
        llm = Tongyi(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            model_name="qwen-turbo",
            temperature=0.7
        )
        
        # 根据意图构建不同的系统提示
        system_prompts = {
            "query_order": "你是一个专业的电商客服，专门处理订单查询问题。请提供准确、友好的回复，指导用户如何查询订单状态。",
            "modify_order": "你是一个电商订单处理专家，帮助用户修改订单信息。请提供清晰的操作指导和注意事项。",
            "cancel_order": "你是一个电商客服，处理订单取消请求。请说明取消流程、退款政策和相关注意事项。",
            "customer_service": "你是一个专业的电商客服代表，提供优质的客户服务。请耐心解答用户问题。",
            "payment_issue": "你是一个电商支付问题处理专家，帮助解决支付相关问题。请提供专业的解决方案。",
            "product_inquiry": "你是一个产品咨询专家，提供详细的商品信息和购买建议。",
            "unknown": "你是一个专业的电商客服助手，请根据用户问题提供合适的帮助。"
        }
        
        system_prompt = system_prompts.get(intent, system_prompts["unknown"])
        
        # 构建完整的提示
        full_prompt = f"""
            {system_prompt}

            用户问题: {user_input}
            识别意图: {intent}

            请提供专业、友好、有帮助的回复：
            """
                    
                    # 调用通义千问生成响应
                    response = llm.invoke(full_prompt).strip()
                    
                    print(f"通义大模型响应: {response}")
                    
                except Exception as e:
                    print(f"通义千问调用失败: {e}")
                    # 降级到预设响应
                    fallback_responses = {
                        "query_order": "您可以通过订单号在'我的订单'页面查询订单状态。如需帮助，请提供订单号。",
                        "modify_order": "订单修改需要在发货前进行。请联系客服或在订单详情页面选择修改选项。",
                        "cancel_order": "未发货订单可以直接取消，已发货订单需要申请退货。取消后款项将在3-5个工作日退回。",
                        "customer_service": "我是您的专属客服助手，很高兴为您服务。请详细描述您遇到的问题。",
                        "payment_issue": "关于支付问题，请提供具体的错误信息，我会帮您解决。",
                        "product_inquiry": "请告诉我您对哪款产品感兴趣，我会为您提供详细信息。",
                        "unknown": "欢迎使用我们的电商服务！我可以帮您处理订单查询、修改、取消等问题。"
                    }
                    response = fallback_responses.get(intent, fallback_responses["unknown"])
                
                return {
                    "response": response,
                    "messages": state.messages + [{"role": "assistant", "content": response}]
                }


def order_processing_node(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """
    订单处理节点
    根据意图执行具体的订单操作
    """
    intent = state.intent
    
    # 模拟订单处理逻辑
    processing_results = {
        "query_order": {
            "action": "查询订单",
            "result": "已为您查询相关订单信息",
            "next_action": "complete"
        },
        "modify_order": {
            "action": "修改订单",
            "result": "订单修改请求已提交，请等待处理",
            "next_action": "complete"
        },
        "cancel_order": {
            "action": "取消订单",
            "result": "订单取消请求已处理",
            "next_action": "complete"
        },
        "customer_service": {
            "action": "客服处理",
            "result": "已转接人工客服",
            "next_action": "transfer_to_human"
        },
        "unknown": {
            "action": "通用处理",
            "result": "已记录您的请求",
            "next_action": "complete"
        }
    }
    
    result = processing_results.get(intent, processing_results["unknown"])
    
    print(f"订单处理结果: {result}")
    
    return {
        "order_info": result,
        "next_action": result["next_action"],
        "messages": state.messages + [{"role": "system", "content": f"处理结果: {result['result']}"}]
    }


def should_continue(state: State) -> Literal["order_processing", "__end__"]:
    """
    条件边：决定是否需要进一步处理
    """
    intent = state.intent
    
    # 所有识别到的意图都需要进入订单处理节点
    if intent in ["query_order", "modify_order", "cancel_order", "customer_service"]:
        return "order_processing"
    
    # 未知意图直接结束
    return "__end__"


def route_after_processing(state: State) -> Literal["tongyi_llm", "__end__"]:
    """
    处理后的路由边：决定下一步操作
    """
    next_action = state.next_action
    
    if next_action == "transfer_to_human":
        return "tongyi_llm"  # 需要大模型生成转接说明
    
    return "__end__"


# 定义工作流图
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("intent_recognition", intent_recognition_node)
    .add_node("tongyi_llm", tongyi_llm_node)
    .add_node("order_processing", order_processing_node)
    .add_edge("__start__", "intent_recognition")
    .add_conditional_edges(
        "intent_recognition",
        should_continue,
        {
            "order_processing": "order_processing",
            "__end__": "__end__"
        }
    )
    .add_conditional_edges(
        "order_processing",
        route_after_processing,
        {
            "tongyi_llm": "tongyi_llm",
            "__end__": "__end__"
        }
    )
    .add_edge("tongyi_llm", "__end__")
    .compile(name="电商订单处理工作流")
)