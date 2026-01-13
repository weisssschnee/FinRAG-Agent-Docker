import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. 页面标题
st.title("💸 智能投研助手 (RAG MVP)")

# 2. 模拟数据 (既然我们解耦了，直接写死在这里最安全)
NEWS_DATA = [
    {"date": "2026-01-12", "content": "美联储宣布降息25个基点，科技股应声大涨。"},
    {"date": "2026-01-11", "content": "特斯拉上海工厂产能利用率突破100%，发布新款人形机器人。"},
    {"date": "2026-01-10", "content": "地缘政治紧张局势升级，原油价格突破90美元。"},
    {"date": "2026-01-09", "content": "DeepSeek发布新一代量化大模型，金融行业震动。"},
    {"date": "2026-01-08", "content": "某大型消费电子公司财报不及预期，股价下跌5%。"}
]

# 3. 初始化模型 (这是最耗时的部分，加个缓存装饰器)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def init_db():
    client = chromadb.Client()
    collection = client.create_collection("financial_news")
    return collection

model = load_model()
collection = init_db()

# 4. 把数据存进去 (只在第一次运行时做)
if collection.count() == 0:
    st.write("正在构建向量知识库...")
    ids = [str(i) for i in range(len(NEWS_DATA))]
    documents = [item["content"] for item in NEWS_DATA]
    metadatas = [{"date": item["date"]} for item in NEWS_DATA]
    
    # 向量化
    embeddings = model.encode(documents).tolist()
    
    # 存入
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    st.success(f"成功存入 {len(NEWS_DATA)} 条新闻！")

# 5. 用户界面
query = st.text_input("请输入问题 (例如: 最近有什么利好?)", "最近有什么利好?")

if st.button("搜索"):
    # 检索
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    
    # 展示结果
    st.subheader("🔍 检索到的相关新闻:")
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        score = results['distances'][0][i]
        
        st.markdown(f"**[{meta['date']}]** {doc}")
        st.caption(f"相似度距离: {score:.4f}")