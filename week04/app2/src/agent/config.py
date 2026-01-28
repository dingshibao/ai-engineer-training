"""
电商订单处理工作流配置
"""

import os
from typing import Dict, Any


class WorkflowConfig:
    """工作流配置类"""
    
    # 通义千问配置
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY", "")
    TONGYI_MODEL = "qwen-turbo"
    TONGYI_MAX_TOKENS = 2000
    TONGYI_TEMPERATURE = 0.7
    
    # 意图识别配置
    INTENT_CONFIDENCE_THRESHOLD = 0.3
    
    # 工作流配置
    MAX_WORKFLOW_STEPS = 10
    WORKFLOW_TIMEOUT = 300  # 5分钟
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # LangSmith配置（用于监控）
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = "ecommerce-order-workflow"
    LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
    
    @classmethod
    def get_tongyi_config(cls) -> Dict[str, Any]:
        """获取通义千问配置"""
        return {
            "api_key": cls.TONGYI_API_KEY,
            "model": cls.TONGYI_MODEL,
            "max_tokens": cls.TONGYI_MAX_TOKENS,
            "temperature": cls.TONGYI_TEMPERATURE
        }
    
    @classmethod
    def get_langsmith_config(cls) -> Dict[str, Any]:
        """获取LangSmith配置"""
        return {
            "api_key": cls.LANGSMITH_API_KEY,
            "project": cls.LANGSMITH_PROJECT,
            "endpoint": cls.LANGSMITH_ENDPOINT
        }


# 意图映射配置
INTENT_MAPPING = {
    "query_order": {
        "name": "订单查询",
        "description": "用户想要查询订单状态、物流信息等",
        "next_nodes": ["tongyi_llm", "order_processing"]
    },
    "modify_order": {
        "name": "订单修改",
        "description": "用户想要修改订单信息",
        "next_nodes": ["order_processing"]
    },
    "cancel_order": {
        "name": "订单取消",
        "description": "用户想要取消订单",
        "next_nodes": ["order_processing"]
    },
    "customer_service": {
        "name": "客服咨询",
        "description": "用户需要人工客服帮助",
        "next_nodes": ["tongyi_llm", "order_processing"]
    },
    "payment_issue": {
        "name": "支付问题",
        "description": "用户遇到支付相关问题",
        "next_nodes": ["tongyi_llm"]
    },
    "product_inquiry": {
        "name": "商品咨询",
        "description": "用户咨询商品信息",
        "next_nodes": ["tongyi_llm"]
    },
    "unknown": {
        "name": "未知意图",
        "description": "无法识别的用户意图",
        "next_nodes": ["tongyi_llm"]
    }
}

# 节点配置
NODE_CONFIG = {
    "intent_recognition": {
        "name": "意图识别",
        "description": "识别用户输入的意图",
        "timeout": 30
    },
    "tongyi_llm": {
        "name": "通义大模型",
        "description": "调用通义千问处理用户问题",
        "timeout": 60
    },
    "order_processing": {
        "name": "订单处理",
        "description": "执行具体的订单操作",
        "timeout": 45
    }
}