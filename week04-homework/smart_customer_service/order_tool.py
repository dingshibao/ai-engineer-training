

from datetime import datetime, timedelta
from typing import List, Dict, Any
from langchain_core.tools import tool
import re
from pydantic import BaseModel
from typing import Optional

@tool
def extract_order_id(query: str) -> str:
    """
    从用户输入中提取订单号，长度不限制。
    支持格式：ZT20250204005678、#ZT20250204005678、或 #数字
    如果提取到订单号，则返回订单号，否则返回None。
    
    Args:
        query: 用户输入
    Returns:
        str | None: 提取到的订单号
        None: 未提取到订单号
    """
    print("-"*50, "extract_order_id", "-"*50)
    # 2 字母 + 数字（如 ZT20250204005678）
    m = re.search(r"[A-Za-z]{2}\d+", query)
    if m:
        return m.group(0).upper()
    # # 前缀 + 数字（如 #123456）
    m = re.search(r"#(\d+)", query)
    if m:
        return m.group(1)
    return "未提取到订单号"

@tool
def get_order_by_id(order_id: str) -> Dict[str, Any] | None:
    """
    根据订单号查询订单
    
    Args:
        order_id: 订单号
    Returns:
        Dict[str, Any] | None: 订单信息
    """
    print("-"*50, "get_order_by_id", "-"*50)
    for order in ORDERS:
        if order.order_id.upper() == order_id.upper():
            return order.model_dump()
    return {
        "order_id": order_id,
        "status": "order_not_found",
        "message": "未找到订单",
    }

# 订单状态枚举
ORDER_STATUS = {
    "pending_pickup": "待揽收",
    "picked_up": "已揽收",
    "in_transit": "运输中",
    "arrived": "已到达",
    "out_for_delivery": "派送中",
    "delivered": "已签收",
    "failed": "派送失败",
    "returned": "已退回",
    "cancelled": "已取消",
}

# 物流公司
COURIERS = ["顺丰速运", "中通快递", "圆通速递", "韵达快递", "申通快递", "极兔速递", "京东物流"]

from pydantic import BaseModel

class OrderInfo(BaseModel):
    order_id: str
    order_no: Optional[str] = None
    status: Optional[str] = None
    status_desc: Optional[str] = None
    courier: Optional[str] = None
    courier_no: Optional[str] = None
    sender: Optional[Dict[str, Any]] = None
    receiver: Optional[Dict[str, Any]] = None

# 预生成的测试数据（固定种子，便于复现）
ORDERS: List[OrderInfo] = [
    OrderInfo(
        order_id="SF20250205001234",
        order_no="SF20250205001234",
        status="in_transit",
        status_desc="运输中",
        courier="顺丰速运",
        courier_no="SF1234567890123",
        sender={"name": "张先生", "phone": "13812345678", "address": "北京市朝阳区建国路88号"},
        receiver={"name": "李女士", "phone": "13987654321", "address": "上海市浦东新区陆家嘴环路1000号"},
    ),
    OrderInfo(
        order_id="ZT20250204005678",
        order_no="ZT20250204005678",
        status="delivered",
        status_desc="已签收",
        courier="中通快递",
        courier_no="ZT9876543210987",
        sender={"name": "王先生", "phone": "13822223333", "address": "广州市天河区体育西路123号"},
        receiver={"name": "刘女士", "phone": "13955556666", "address": "成都市武侯区天府大道789号"},
    ),
    OrderInfo(
        order_id="YT20250203009123",
        order_no="YT20250203009123",
        status="pending_pickup",
        status_desc="待揽收",
        courier="圆通速递",
        courier_no="YT4567890123456",
        sender={"name": "陈先生", "phone": "13877778888", "address": "深圳市南山区科技园南路66号"},
        receiver={"name": "赵女士", "phone": "13911112222", "address": "武汉市洪山区光谷大道321号"},
    ),
    OrderInfo(
        order_id="YD20250202003456",
        order_no="YD20250202003456",
        status="out_for_delivery",
        status_desc="派送中",
        courier="韵达快递",
        courier_no="YD7890123456789",
        sender={"name": "周先生", "phone": "13833334444", "address": "杭州市西湖区文三路456号"},
        receiver={"name": "吴女士", "phone": "13966667777", "address": "南京市鼓楼区中山北路654号"},
    ),
    OrderInfo(
        order_id="JD20250201007890",
        order_no="JD20250201007890",
        status="failed",
        status_desc="派送失败",
        courier="京东物流",
        courier_no="JD3210987654321",
        sender={"name": "郑先生", "phone": "13899990000", "address": "北京市朝阳区建国路88号"},
        receiver={"name": "孙女士", "phone": "13944445555", "address": "西安市雁塔区高新路987号"},
    ),
    OrderInfo(
        order_id="ST20250131001111",
        order_no="ST20250131001111",
        status="returned",
        status_desc="已退回",
        courier="申通快递",
        courier_no="ST6543210987654",
        sender={"name": "林先生", "phone": "13888889999", "address": "上海市浦东新区陆家嘴环路1000号"},
        receiver={"name": "黄女士", "phone": "13922223333", "address": "重庆市渝中区解放碑步行街"},
    ),
    OrderInfo(
        order_id="JT20250205002222",
        order_no="JT20250205002222",
        status="arrived",
        status_desc="已到达",
        courier="极兔速递",
        courier_no="JT1122334455667",
        sender={"name": "李女士", "phone": "13866667777", "address": "广州市天河区体育西路123号"},
        receiver={"name": "张先生", "phone": "13988889999", "address": "深圳市南山区科技园南路66号"},
    ),
    OrderInfo(
        order_id="SF20250204003333",
        order_no="SF20250204003333",
        status="picked_up",
        status_desc="已揽收",
        courier="顺丰速运",
        courier_no="SF7788990011223",
        sender={"name": "王女士", "phone": "13855556666", "address": "成都市武侯区天府大道789号"},
        receiver={"name": "刘先生", "phone": "13911112222", "address": "杭州市西湖区文三路456号"},
    ),
    OrderInfo(
        order_id="ZT20250130004444",
        order_no="ZT20250130004444",
        status="cancelled",
        status_desc="已取消",
        courier="中通快递",
        courier_no="ZT2233445566778",
        sender={"name": "陈女士", "phone": "13844445555", "address": "武汉市洪山区光谷大道321号"},
        receiver={"name": "赵先生", "phone": "13933334444", "address": "南京市鼓楼区中山北路654号"},
    ),
]