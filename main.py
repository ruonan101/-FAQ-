from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
import os
import json

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化sentence transformer模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 存储问答对和嵌入向量
faqs = []
embeddings = []

class FAQ(BaseModel):
    question: str
    answer: str

class QueryResult(BaseModel):
    question: str
    answer: str
    score: float

class RelatedQuestion(BaseModel):
    question: str
    score: float

def save_faqs():
    """保存FAQ到文件"""
    with open('faqs.json', 'w', encoding='utf-8') as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)

def load_faqs():
    """从文件加载FAQ"""
    global faqs, embeddings
    if os.path.exists('faqs.json'):
        with open('faqs.json', 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        # 重建嵌入向量
        embeddings = [model.encode(faq['question']) for faq in faqs]

# 启动时加载FAQ
load_faqs()

@app.get("/")
async def read_root():
    """返回主页"""
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
        return HTMLResponse(content=content)

@app.post("/add_faq")
async def add_faq(faq: FAQ):
    """添加新的FAQ"""
    # 生成问题的嵌入向量
    embedding = model.encode(faq.question)
    
    # 添加到列表
    faqs.append({"question": faq.question, "answer": faq.answer})
    embeddings.append(embedding)
    
    # 保存到文件
    save_faqs()
    
    return {"status": "success", "message": "FAQ added successfully"}

@app.get("/related_questions")
async def get_related_questions(question: str, min_score: float = 0.5, limit: int = 5) -> List[RelatedQuestion]:
    """获取相关问题列表"""
    if len(faqs) == 0:
        return []
        
    # 生成查询问题的嵌入向量
    query_embedding = model.encode(question)
    
    # 计算与所有FAQ的余弦相似度
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # 获取相似度排名前N的问题
    related_indices = np.argsort(similarities)[::-1][:limit]
    related_questions = []
    
    for idx in related_indices:
        score = similarities[idx]
        if score >= min_score:
            related_questions.append(
                RelatedQuestion(
                    question=faqs[idx]["question"],
                    score=float(score)
                )
            )
    
    return related_questions

@app.get("/query")
async def query_faq(question: str, threshold: float = 0.7) -> Optional[QueryResult]:
    """查询FAQ"""
    if len(faqs) == 0:
        return None
        
    # 生成查询问题的嵌入向量
    query_embedding = model.encode(question)
    
    # 计算与所有FAQ的余弦相似度
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # 找到最相似的FAQ
    max_sim_idx = np.argmax(similarities)
    max_sim = similarities[max_sim_idx]
    
    if max_sim < threshold:
        return None
        
    matched_faq = faqs[max_sim_idx]
    return QueryResult(
        question=matched_faq["question"],
        answer=matched_faq["answer"],
        score=float(max_sim)
    )

@app.get("/list_faqs")
async def list_faqs():
    """列出所有FAQ"""
    return faqs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
