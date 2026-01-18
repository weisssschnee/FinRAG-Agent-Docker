from sentence_transformers import SentenceTransformer
# 这会在你本地下载模型，通常在 C:\Users\你的用户名\.cache\torch\sentence_transformers\
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save("local_model")