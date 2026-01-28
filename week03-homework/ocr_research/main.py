from llama_index.core.readers.base import BaseReader

from llama_index.core.schema import Document
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from paddleocr import PaddleOCR
from llama_index.llms.openai_like import OpenAILike
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 设置 embed_model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    trust_remote_code=False,
)

# 设置 LLM（如果 API key 存在）
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_api_key:
    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key=deepseek_api_key,
        is_chat_model=True
    )
else:
    print("Warning: DEEPSEEK_API_KEY not found. LLM features will not be available.")
    print("Please set DEEPSEEK_API_KEY in .env file or environment variables.")

class ImageOCRReader(BaseReader):
    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        self.lang = lang
        self.use_gpu = use_gpu
        # 为了性能，在初始化时加载模型（旧版本 API）
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, **kwargs)

    def load_data(self, file: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串、Path 对象或路径列表
        Returns:
            List[Document]
        """
        # 处理 file 参数，统一转换为列表格式
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file

        documents = []
        for image_path in files:
            image_path_str = str(image_path)
            # 使用 PaddleOCR 提取文本（旧版本 API：ocr.ocr()）
            result = self._ocr.ocr(image_path_str, cls=True)

            if not result or not result[0]:
                # 如果 OCR 未返回任何结果，则跳过此图像
                print(f"Warning: No text detected in {image_path_str}")
                continue

            text_lines = []
            confidences = []
            
            # 遍历所有检测到的文本行
            for line in result[0]:
                text = line[1][0]  # 提取文本内容
                confidence = line[1][1]  # 提取置信度
                text_lines.append(text)
                confidences.append(confidence)

            # 将所有文本行拼接在一起，使用换行符分隔
            full_text = "\n".join(text_lines)
            
            # 计算平均置信度
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # 构造 Document 对象
            doc = Document(
                text=full_text,
                metadata={
                    "image_path": image_path_str,
                    "ocr_model": "PP-OCRv4",
                    "language": self.lang,
                    "num_text_lines": len(text_lines),
                    "avg_confidence": float(avg_confidence)
                }
            )
            documents.append(doc)

        return documents

def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m ocr_research.main 运行
    reader = ImageOCRReader(lang='ch', use_gpu=False)

    parent_dir = Path(__file__).parent.parent
    
    files = [parent_dir / 'ocr_test1_ticket.png', parent_dir / 'ocr_test2_table.png', parent_dir / 'ocr_test3_car.jpg']
    documents = reader.load_data(files)
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("我想要知道图片中机票的信息")
    print(response)

if __name__ == "__main__":
    main()