# pip install langchain_community datasets ragas

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from ragas.metrics import context_recall, context_precision

data_samples = {
    'question': [
        '杭州最值得去的景点有哪些？',
        '去成都旅游的话，有哪些特色美食推荐？',
        '在西安游览时，参观兵马俑需要提前预约吗？'
    ],
    'answer': [
        '杭州西湖、灵隐寺和千岛湖是比较受欢迎的景点。',
        '成都有很多好吃的，比如火锅、串串香和担担面。',
        '参观兵马俑不需要预约，现场买票就可以进去。'
    ],
    'ground_truth': [
        '杭州必游景点包括西湖、灵隐寺、雷峰塔、千岛湖和宋城，其中西湖是国家5A级景区，建议清晨游览以避开人流。',
        '成都作为美食之都，推荐品尝火锅、串串香、担担面、龙抄手和钟水饺，宽窄巷子和锦里是集中体验地道小吃的好去处。',
        '参观秦始皇兵马俑博物馆必须通过官方平台提前实名预约购票，旺季时需至少提前3天预约，现场不保证有票。'
    ],
    'contexts': [
    # ...
    [
        '西湖是杭州的核心景区，国家5A级旅游景点，以湖光山色和历史文化闻名。',
        '灵隐寺是中国著名的佛教古刹，位于杭州西湖区。',
        '千岛湖以湖泊和岛屿景观著称，适合度假与户外活动。'
    ],
    [
        '成都被誉为“美食之都”，火锅是其代表性饮食，麻辣鲜香。',
        '串串香起源于四川街头，食材丰富，可自选。',
        '担担面是川菜经典面食，以红油和肉末调味。'
    ],
    [
        '秦始皇兵马俑位于陕西省西安市临潼区，是世界文化遗产。',
        '兵马俑博物馆实行实名制预约购票制度，建议提前在官网或官方公众号预约。',
        '旺季参观需至少提前3天预约，避免现场无票。'
    ]
]
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(
    dataset=dataset,
    metrics=[context_recall, context_precision],
    llm=Tongyi(model_name="qwen-plus"),
    embeddings=DashScopeEmbeddings(model="text-embedding-v3")
)
print(score.to_pandas())