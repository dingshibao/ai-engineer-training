"""
电商订单处理相关的数据模型
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class OrderInfo(BaseModel):
    """订单信息模型"""
    order_id: str = Field(..., description="订单ID")
    user_id: str = Field(..., description="用户ID")
    status: str = Field(..., description="订单状态")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="订单商品")
    total_amount: float = Field(..., description="订单总金额")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class IntentResult(BaseModel):
    """意图识别结果模型"""
    intent: str = Field(..., description="识别到的意图")
    confidence: float = Field(..., description="置信度")
    entities: Dict[str, Any] = Field(default_factory=dict, description="提取的实体")


class LLMResponse(BaseModel):
    """大模型响应模型"""
    content: str = Field(..., description="响应内容")
    model: str = Field(default="tongyi", description="使用的模型")
    tokens_used: int = Field(default=0, description="使用的token数量")
    response_time: float = Field(default=0.0, description="响应时间（秒）")


class ProcessingResult(BaseModel):
    """处理结果模型"""
    success: bool = Field(..., description="处理是否成功")
    action: str = Field(..., description="执行的操作")
    result: str = Field(..., description="处理结果描述")
    data: Optional[Dict[str, Any]] = Field(default=None, description="相关数据")
    next_action: str = Field(default="complete", description="下一步操作")


class WorkflowState(BaseModel):
    """工作流状态模型"""
    session_id: str = Field(..., description="会话ID")
    user_input: str = Field(..., description="用户输入")
    intent_result: Optional[IntentResult] = Field(default=None, description="意图识别结果")
    llm_response: Optional[LLMResponse] = Field(default=None, description="大模型响应")
    processing_result: Optional[ProcessingResult] = Field(default=None, description="处理结果")
    current_step: str = Field(default="start", description="当前步骤")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")