import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from typing import List, Optional
from pydantic import BaseModel
import csv
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

class FAQ:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer
        self.embedding = None

class FAQSystem:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.faqs = []
        self.load_faqs()
        
    def load_faqs(self):
        # 从CSV文件加载FAQ
        if os.path.exists('sample_faqs.csv'):
            with open('sample_faqs.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 2:
                        faq = FAQ(row[0], row[1])
                        faq.embedding = self.model.encode(row[0], convert_to_tensor=True)
                        self.faqs.append(faq)
            print(f"已加载 {len(self.faqs)} 个FAQ问题")
        else:
            print("警告: sample_faqs.csv 文件不存在")

    def find_similar(self, query: str, threshold: float = 0.3) -> Optional[dict]:
        if not self.faqs:
            return None
            
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # 计算相似度
        similarities = []
        for faq in self.faqs:
            similarity = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                faq.embedding.cpu().numpy().reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # 找到最相似的FAQ
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity >= threshold:
            return {
                "question": self.faqs[max_similarity_idx].question,
                "answer": self.faqs[max_similarity_idx].answer,
                "score": float(max_similarity)
            }
        return None

    def get_all_faqs(self) -> List[dict]:
        return [{"question": faq.question, "answer": faq.answer} for faq in self.faqs]

    def find_related(self, query: str, limit: int = 5) -> List[dict]:
        if not self.faqs:
            return []
            
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # 计算所有FAQ的相似度
        similarities = []
        for faq in self.faqs:
            similarity = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                faq.embedding.cpu().numpy().reshape(1, -1)
            )[0][0]
            similarities.append((similarity, faq))
        
        # 按相似度排序并返回前N个结果
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            {"question": faq.question, "answer": faq.answer, "score": float(score)}
            for score, faq in similarities[:limit]
        ]

# 创建FAQ系统实例
faq_system = FAQSystem()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/api/query")
async def query_faq(q: str):
    result = faq_system.find_similar(q)
    if result:
        return result
    raise HTTPException(status_code=404, detail="未找到相关答案")

@app.get("/api/list")
async def list_faqs():
    return faq_system.get_all_faqs()

@app.get("/api/related")
async def get_related(q: str, limit: int = 5):
    return faq_system.find_related(q, limit)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
