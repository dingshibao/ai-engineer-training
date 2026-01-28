import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness

print("开始创建测试数据...")

# 简化的测试数据
data_samples = {
    'question': ['杭州最值得去的景点有哪些？'],
    'answer': ['杭州西湖、灵隐寺和千岛湖是比较受欢迎的景点。'],
    'ground_truth': ['杭州必游景点包括西湖、灵隐寺、雷峰塔、千岛湖和宋城，其中西湖是国家5A级景区，建议清晨游览以避开人流。']
}

print("创建数据集...")
dataset = Dataset.from_dict(data_samples)

print("初始化模型...")
llm = Tongyi(model_name="qwen-plus")
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

print("开始评估...")
try:
    score = evaluate(
        dataset=dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=embeddings
    )
    print("评估完成！")
    print(score.to_pandas())
except Exception as e:
    print(f"评估过程中出现错误: {e}")
    import traceback
    traceback.print_exc()