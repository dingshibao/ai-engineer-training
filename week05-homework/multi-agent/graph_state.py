from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class ArticleState(BaseModel):
    # 对话历史
    messages: list[BaseMessage] = Field(default_factory=list)

    # 用户输入
    user_input: str = ""

    # 研究数据
    research_data: str = ""

    # 文章草稿
    article_draft: str = ""

    # 文章审核结果
    article_review_result: str = ""

    # 文章润色结果
    article_polish_result: str = ""

    # write_retry: 重试次数
    write_retry_count: int = 0

    # 文章风格
    article_style: str = ""

    # 文章长度
    article_length: int = 0

