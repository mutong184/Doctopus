from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbeddings:
    def __init__(self, model_name: str = "/home/lijiajun/RAG/facebook/contriever", use_gpu: bool = True):
        """
        初始化 HuggingFace 的嵌入生成器。
        :param model_name: 使用的预训练模型名称。
        :param use_gpu: 是否使用 GPU 加速。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 选择设备
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # 将模型加载到指定设备

    def embed_text(self, text: str) -> torch.Tensor:
        """
        对单个文本生成嵌入向量。
        :param text: 输入文本。
        :return: 文本的嵌入向量。
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取池化的嵌入（这里使用平均池化）
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return embedding.cpu()  # 将结果转回 CPU 以便后续使用

    def embed_documents(self, texts: list[str]) -> list[torch.Tensor]:
        """
        对多个文本生成嵌入向量。
        :param texts: 输入文本列表。
        :return: 文本嵌入向量列表。
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings
    

def get_similiar_chunks(file,pencent):
    # 打开一个长文本文件并读取内容
    try:
        with open(file, encoding='utf-8') as f:
            state_of_the_union = f.read()
        
    except UnicodeDecodeError as e:
        # 捕获编码异常并输出文件名
        print(f"Error reading file {file}")
        return ""

    embeddings = HuggingFaceEmbeddings()

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=pencent * 100)

    docs = text_splitter.create_documents([state_of_the_union])

    list_chunks = [doc.page_content for doc in docs]

    return list_chunks



