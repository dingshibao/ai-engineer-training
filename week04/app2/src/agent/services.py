"""
电商订单处理服务层
包含通义大模型调用、意图识别等服务
"""

import os
import json
import time
from typing import Dict, Any, Optional
from .models import IntentResult, LLMResponse, OrderInfo


class IntentRecognitionService:
    """意图识别服务"""
    
    def __init__(self):
        self.intent_keywords = {
            "query_order": ["查询", "查看", "订单状态", "物流", "配送", "到哪了"],
            "modify_order": ["修改", "更改", "变更", "地址", "电话", "收货"],
            "cancel_order": ["取消", "退单", "撤销", "不要了", "退货"],
            "customer_service": ["投诉", "问题", "客服", "人工", "帮助", "咨询"],
            "payment_issue": ["支付", "付款", "扣款", "退款", "发票"],
            "product_inquiry": ["商品", "产品", "规格", "参数", "介绍"]
        }
    
    def recognize_intent(self, user_input: str) -> IntentResult:
        """
        识别用户输入的意图
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            IntentResult: 意图识别结果
        """
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # 计算每个意图的匹配分数
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        if not intent_scores:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                entities={}
            )
        
        # 选择得分最高的意图
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        # 提取实体（简单实现）
        entities = self._extract_entities(user_input, best_intent)
        
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities
        )
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """提取实体信息"""
        entities = {}
        
        # 简单的实体提取逻辑
        if intent == "query_order":
            # 尝试提取订单号
            import re
            order_pattern = r'[A-Z0-9]{10,20}'
            matches = re.findall(order_pattern, text)
            if matches:
                entities["order_id"] = matches[0]
        
        return entities


class TongyiLLMService:
    """通义千问大模型服务"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
    def generate_response(self, prompt: str, intent: str = "") -> LLMResponse:
        """
        生成大模型响应
        
        Args:
            prompt: 输入提示
            intent: 用户意图
            
        Returns:
            LLMResponse: 大模型响应结果
        """
        start_time = time.time()
        
        # 根据意图构建系统提示
        system_prompts = {
            "query_order": "你是一个专业的电商客服，专门处理订单查询问题。请提供准确、友好的回复。",
            "modify_order": "你是一个电商订单处理专家，帮助用户修改订单信息。请提供清晰的操作指导。",
            "cancel_order": "你是一个电商客服，处理订单取消请求。请说明取消流程和相关政策。",
            "customer_service": "你是一个专业的电商客服代表，提供优质的客户服务。",
            "payment_issue": "你是一个电商支付问题处理专家，帮助解决支付相关问题。",
            "product_inquiry": "你是一个产品咨询专家，提供详细的商品信息和建议。"
        }
        
        system_prompt = system_prompts.get(intent, "你是一个专业的电商客服助手。")
        
        # 调用真实的通义千问API
        try:
            from langchain_community.llms import Tongyi
            
            llm = Tongyi(
                dashscope_api_key=self.api_key,
                model_name="qwen-turbo",
                temperature=0.7
            )
            
            full_prompt = system_prompt + "\n\n用户问题: " + prompt + "\n\n请提供专业回复："
            response_content = llm.invoke(full_prompt).strip()
            
        except Exception as e:
            print(f"通义千问API调用失败: {e}")
            response_content = self._generate_mock_response(prompt, intent)
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=response_content,
            model="qwen-turbo",
            tokens_used=len(prompt) + len(response_content),
            response_time=response_time
        )
    
    def _mock_api_call(self, prompt: str, system_prompt: str) -> str:
        """模拟API调用"""
        # 实际实现中，这里会调用通义千问的API
        return f"基于系统提示'{system_prompt}'对'{prompt}'的专业回复"
    
    def _generate_mock_response(self, prompt: str, intent: str) -> str:
        """生成模拟响应"""
        mock_responses = {
            "query_order": "您好！我来帮您查询订单状态。请提供您的订单号，我会立即为您查询最新的物流信息。",
            "modify_order": "我理解您需要修改订单信息。请注意，订单修改需要在发货前进行。您可以在'我的订单'页面找到相应订单，点击'修改订单'按钮进行操作。",
            "cancel_order": "关于订单取消，我来为您说明：未发货的订单可以直接取消，已发货的订单需要申请退货。取消后，款项将在3-5个工作日内退回到您的原支付账户。",
            "customer_service": "您好！我是您的专属客服助手，很高兴为您服务。请详细描述您遇到的问题，我会尽力为您解决。",
            "payment_issue": "关于支付问题，我来帮您处理。请告诉我具体遇到了什么支付问题，比如支付失败、重复扣款或退款问题等。",
            "product_inquiry": "我很乐意为您介绍我们的产品。请告诉我您对哪款产品感兴趣，我会为您提供详细的产品信息和购买建议。"
        }
        
        return mock_responses.get(intent, "感谢您的咨询，我会尽力为您提供帮助。请详细描述您的需求。")


class OrderService:
    """订单服务"""
    
    def __init__(self):
        # 模拟订单数据库
        self.orders = {}
    
    def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """获取订单信息"""
        return self.orders.get(order_id)
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """更新订单信息"""
        if order_id in self.orders:
            order = self.orders[order_id]
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            return True
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            return True
        return False