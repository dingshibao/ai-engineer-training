# 进入代码目录
```
cd week01
```

# 如何安装依赖
不管使用conda与否，我们都统一用uv来做依赖安装，一是因为它速度最快，二是因为比较好管理安装源
## 使用Conda的看这里
```
conda create --name ai-engineer-week01 python=3.11
conda activate ai-engineer-week01
pip install uv
uv pip install -r requirements.txt
```

## 不想使用Conda，想直接使用uv的看这里
```
pip install uv
uv sync --locked
source .venv/bin/activate 
```

# 设置环境变量
```
cp .env.example .env
```
然后打开.env，将里面的API KEY替换为自己的KEY，您可以访问：https://api.vveai.com 来获取Key，或使用其他自己喜欢的模型供应商，比如: https://openrouter.ai
```
OPENAI_API_KEY=replace-me-to-your-own-key>
```