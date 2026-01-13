import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ========== 配置 ==========
CSV_PATH = "news_data.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 轻量模型，Docker里跑得动
TOP_K = 3

# ========== 1. 读取CSV ==========
df = pd.read_csv(CSV_PATH)

documents = df["content"].tolist()
metadatas = [{"title": t} for t in df["title"].tolist()]
ids = [str(i) for i in df["id"].tolist()]

# ========== 2. 加载Embedding模型 ==========
print(">> Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

embeddings = model.encode(documents).tolist()

# ========== 3. 初始化 ChromaDB ==========
print(">> Initializing ChromaDB...")
client = chromadb.Client()
collection = client.get_or_create_collection("news_rag")

# 防止重复插入
collection.delete(where={})

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings
)

print(">> Vector DB loaded successfully!")

# ========== 4. 查询 ==========
query = "最近有什么利好？"
print(f"\n>> Query: {query}")

query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=TOP_K
)

# ========== 5. 输出结果 ==========
print("\n>> Top Results:")
for i in range(TOP_K):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    score = results["distances"][0][i]
    print(f"\n[{i+1}] 标题: {meta['title']}")
    print(f"内容: {doc}")
    print(f"距离分数: {score:.4f}")
