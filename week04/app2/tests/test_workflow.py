"""
电商订单处理工作流测试
"""

import pytest
import sys
import os
import asyncio

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import graph, State, Context
from src.agent.services import IntentRecognitionService, TongyiLLMService


class TestOrderWorkflow:
    """订单工作流测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.workflow = graph
        self.intent_service = IntentRecognitionService()
        self.llm_service = TongyiLLMService()
    
    def test_intent_recognition_service(self):
        """测试意图识别服务"""
        test_cases = [
            ("我想查询订单状态", "query_order"),
            ("我要取消订单", "cancel_order"),
            ("修改收货地址", "modify_order"),
            ("需要客服帮助", "customer_service"),
            ("随便说点什么", "unknown")
        ]
        
        for user_input, expected_intent in test_cases:
            result = self.intent_service.recognize_intent(user_input)
            assert result.intent == expected_intent
            assert result.confidence >= 0.0
    
    def test_tongyi_llm_service(self):
        """测试通义大模型服务"""
        test_prompts = [
            ("查询订单", "query_order"),
            ("取消订单", "cancel_order"),
            ("客服咨询", "customer_service")
        ]
        
        for prompt, intent in test_prompts:
            response = self.llm_service.generate_response(prompt, intent)
            assert response.content is not None
            assert len(response.content) > 0
            assert response.model == "qwen-turbo"
            assert response.response_time >= 0
    
    def test_workflow_query_order(self):
        """测试订单查询工作流"""
        initial_state = State(
            user_input="我想查询我的订单状态",
            intent="",
            order_info={},
            response="",
            next_action="",
            messages=[]
        )
        
        context = Context(user_id="test_user", session_id="test_session")
        
        # 使用同步调用
        result = self.workflow.invoke(initial_state, {"configurable": context})
        
        assert result["intent"] == "query_order"
        assert result["response"] is not None
        assert len(result["messages"]) > 0
    
    def test_workflow_cancel_order(self):
        """测试订单取消工作流"""
        initial_state = State(
            user_input="我要取消刚才下的订单",
            intent="",
            order_info={},
            response="",
            next_action="",
            messages=[]
        )
        
        context = Context(user_id="test_user", session_id="test_session")
        
        result = self.workflow.invoke(initial_state, {"configurable": context})
        
        assert result["intent"] == "cancel_order"
        assert result["order_info"]["action"] == "取消订单"
        assert result["next_action"] == "complete"
    
    def test_workflow_customer_service(self):
        """测试客服咨询工作流"""
        initial_state = State(
            user_input="订单有问题，需要客服帮助",
            intent="",
            order_info={},
            response="",
            next_action="",
            messages=[]
        )
        
        context = Context(user_id="test_user", session_id="test_session")
        
        result = self.workflow.invoke(initial_state, {"configurable": context})
        
        assert result["intent"] == "customer_service"
        assert result["order_info"]["next_action"] == "transfer_to_human"
    
    def test_workflow_unknown_intent(self):
        """测试未知意图工作流"""
        initial_state = State(
            user_input="今天天气怎么样",
            intent="",
            order_info={},
            response="",
            next_action="",
            messages=[]
        )
        
        context = Context(user_id="test_user", session_id="test_session")
        
        result = self.workflow.invoke(initial_state, {"configurable": context})
        
        assert result["intent"] == "unknown"
        assert result["response"] is not None
    
    def test_multiple_workflow_runs(self):
        """测试多次工作流运行"""
        test_inputs = [
            "查询订单状态",
            "修改收货地址",
            "取消订单",
            "需要客服帮助"
        ]
        
        context = Context(user_id="test_user", session_id="test_session")
        
        for user_input in test_inputs:
            initial_state = State(
                user_input=user_input,
                intent="",
                order_info={},
                response="",
                next_action="",
                messages=[]
            )
            
            result = self.workflow.invoke(initial_state, {"configurable": context})
            
            # 验证基本结果
            assert result["intent"] is not None
            assert result["intent"] != ""
            assert len(result["messages"]) > 0


def test_simple_intent_recognition():
    """简单的意图识别测试（同步）"""
    service = IntentRecognitionService()
    result = service.recognize_intent("我想查询订单")
    assert result.intent == "query_order"
    assert result.confidence > 0


def test_simple_llm_service():
    """简单的LLM服务测试（同步）"""
    service = TongyiLLMService()
    response = service.generate_response("测试", "query_order")
    assert response.content is not None
    assert len(response.content) > 0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])