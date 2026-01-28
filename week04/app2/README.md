# 电商订单处理工作流

基于LangGraph构建的电商订单处理工作流，集成通义千问大模型，支持意图识别和条件路由。

## 功能特性

- **意图识别**: 自动识别用户的订单相关意图（查询、修改、取消、客服咨询等）
- **通义大模型集成**: 调用通义千问处理复杂的用户问题
- **条件路由**: 根据意图和处理结果智能路由到不同的处理节点
- **工程化设计**: 完整的项目结构，包含服务层、模型定义和测试用例
- **LangSmith监控**: 支持LangSmith监控工作流运行过程

## 项目结构

```
app2/
├── src/
│   └── agent/
│       ├── __init__.py          # 模块初始化
│       ├── graph.py             # 工作流图定义
│       ├── models.py            # 数据模型
│       ├── services.py          # 服务层
│       └── config.py            # 配置文件
├── tests/
│   └── test_workflow.py         # 测试用例
└── README.md                    # 项目说明
```

## 工作流架构

### 节点说明

1. **意图识别节点** (`intent_recognition_node`)
   - 分析用户输入，识别订单相关意图
   - 支持的意图类型：查询订单、修改订单、取消订单、客服咨询等

2. **通义大模型节点** (`tongyi_llm_node`)
   - 调用通义千问生成专业的客服回复
   - 根据不同意图使用相应的提示模板

3. **订单处理节点** (`order_processing_node`)
   - 执行具体的订单操作
   - 返回处理结果和下一步操作建议

### 边的逻辑

1. **条件边** (`should_continue`)
   - 根据意图识别结果决定是否需要进一步处理
   - 客服相关问题会路由到订单处理节点

2. **处理后路由边** (`route_after_processing`)
   - 根据处理结果决定下一步操作
   - 需要人工转接时会再次调用大模型生成说明

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install langgraph langsmith pydantic langchain-community dashscope

# 设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export LANGSMITH_API_KEY="your_langsmith_api_key"

# 或者创建 .env 文件
cp .env.example .env
# 然后编辑 .env 文件，填入您的API密钥
```

#### 获取通义千问API密钥

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通DashScope服务
4. 创建API密钥
5. 将密钥配置到环境变量 `DASHSCOPE_API_KEY`

### 2. 运行工作流

```python
from src.agent.graph import get_compiled_workflow

# 创建工作流实例
app = get_compiled_workflow()

# 定义初始状态
initial_state = {
    "messages": [],
    "user_input": "我想查询我的订单状态",
    "intent": "",
    "order_info": {},
    "response": "",
    "next_action": ""
}

# 运行工作流
result = app.invoke(initial_state)
print(result)
```

### 3. 运行测试

```bash
# 运行单元测试
python -m pytest tests/test_workflow.py -v

# 运行工作流演示
python src/agent/graph.py
```

## 支持的意图类型

| 意图类型 | 关键词示例 | 处理方式 |
|---------|-----------|----------|
| query_order | 查询、查看、订单状态 | 引导用户提供订单号进行查询 |
| modify_order | 修改、更改、地址 | 提供订单修改流程指导 |
| cancel_order | 取消、退单、撤销 | 说明取消政策和退款流程 |
| customer_service | 投诉、问题、客服 | 转接人工客服或提供专业回复 |
| payment_issue | 支付、付款、退款 | 处理支付相关问题 |
| product_inquiry | 商品、产品、规格 | 提供产品信息和购买建议 |

## 配置说明

### 通义千问配置

在 `config.py` 中配置通义千问相关参数：

```python
DASHSCOPE_API_KEY = "your_api_key"
TONGYI_MODEL = "qwen-turbo"
TONGYI_MAX_TOKENS = 2000
TONGYI_TEMPERATURE = 0.7
```

**重要说明**：
- 意图识别节点使用通义千问进行智能意图分析
- 大模型节点调用真实的通义千问API生成回复
- 如果API调用失败，会自动降级到预设回复
- 确保设置了正确的 `DASHSCOPE_API_KEY` 环境变量

### LangSmith监控配置

```python
LANGSMITH_API_KEY = "your_langsmith_api_key"
LANGSMITH_PROJECT = "ecommerce-order-workflow"
```

## 扩展开发

### 添加新的意图类型

1. 在 `services.py` 的 `IntentRecognitionService` 中添加新的关键词
2. 在 `graph.py` 中更新相应的处理逻辑
3. 在 `config.py` 中添加意图映射配置

### 集成通义千问API

在 `services.py` 的 `TongyiLLMService` 中实现API调用：

```python
def _mock_api_call(self, prompt: str, system_prompt: str) -> str:
    # 实现真实的通义千问API调用
    # 参考阿里云DashScope文档
    pass
```
