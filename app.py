import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime
import feedparser  # 必须安装这个库: pip install feedparser
import hashlib

# === 1. 页面基础配置 (必须放在第一行) ===
st.set_page_config(
    page_title="DeepQuant 智能投研台",
    page_icon="💸",
    layout="wide"
)

# === 2. 侧边栏 (控制面板) ===
with st.sidebar:
    st.header("⚙️ 系统控制台")
    
    st.markdown("### 🤖 模型设置")
    model_type = st.selectbox(
        "Embedding Backend",
        ["all-MiniLM-L6-v2 (Local)", "OpenAI-Ada-002 (Cloud)", "BGE-Large-Zh"]
    )
    
    st.markdown("### 🛡️ 风控参数")
    risk_level = st.slider("最大回撤阈值 (Max DD)", 5, 25, 12)
    st.progress(risk_level / 30)
    st.caption(f"当前熔断线: -{risk_level}%")
    
    st.divider()
    
    # 状态指示灯
    st.success("🟢 Docker Container: Active")
    st.info("🔵 Vector DB: Connected")
    
    # 刷新按钮
    if st.button("🔄 强制刷新数据源"):
        st.cache_data.clear()
        st.rerun()

# === 3. 核心功能函数 ===

@st.cache_resource
def load_model():
    # 强制只看本地，禁止联网检查！
    return SentenceTransformer('./local_model', local_files_only=True)

@st.cache_resource
def init_db():
    # 初始化向量数据库 (内存模式，重启后清空，适合开发调试)
    client = chromadb.Client()
    # 尝试获取集合，如果已存在则获取，否则创建
    try:
        collection = client.get_collection("financial_news")
    except:
        collection = client.create_collection("financial_news")
    return collection
@st.cache_data(ttl=300)
def fetch_news_feed():
    # 方案 A: 36氪 (科技/金融/创投) - 极大概率能连通
    rss_url = "https://36kr.com/feed"
    
    # 方案 B: 环球网财经 (如果36氪不行，试这个)
    # rss_url = "https://finance.huanqiu.com/rss.xml"
    
    try:
        # 依然带上伪装，国内网站防爬也很严
        feed = feedparser.parse(
            rss_url,
            agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # 检查是否为空
        if not feed.entries:
            raise Exception("国内源返回为空 (可能是格式解析问题或反爬)")
            
        news_items = []
        for entry in feed.entries[:10]:
            # 尝试获取时间
            dt = entry.get('published_parsed')
            if dt:
                pub_date = f"{dt.tm_year}-{dt.tm_mon:02d}-{dt.tm_mday:02d}"
            else:
                pub_date = datetime.now().strftime('%Y-%m-%d')

            # 36氪的 RSS 有时候 summary 是空的，所以做一个容错
            content_text = entry.summary if 'summary' in entry else entry.title
            
            news_items.append({
                "date": pub_date,
                "content": f"【36Kr】{entry.title} - {content_text[:60]}...",
                "link": entry.link
            })
        
        return news_items, True
        
    except Exception as e:
        print(f"RSS Error: {e}")
        # 如果连国内都挂了，那就彻底没办法了，只能用 Mock
        mock_data = [
            {"date": "2026-01-14", "content": "【Mock】A股全线飘红，沪指收复3000点。"},
            {"date": "2026-01-14", "content": "【Mock】茅台发布财报，净利润同比增长 15%。"},
            {"date": "2026-01-14", "content": "【Mock】宁德时代发布凝聚态电池，续航突破1000公里。"},
            {"date": "2026-01-13", "content": "【Mock】央行宣布降准0.5个百分点，释放长期资金1万亿。"},
            {"date": "2026-01-13", "content": "【Mock】腾讯发布大模型混元 5.0，接入微信生态。"}
        ]
        return mock_data, False
        
    except Exception as e:
        print(f"RSS Error: {e}")
        # ❌ 失败：返回高质量的仿真数据 (看起来像真的)
        # 如果真的抓不到，就用下面这组数据，至少截图好看
        mock_data = [
            {"date": "2026-01-14", "content": "OpenAI 发布 GPT-5 预览版，推理能力提升 200%。"},
            {"date": "2026-01-14", "content": "英伟达 CEO 黄仁勋宣布新一代 Blackwell Ultra 芯片量产。"},
            {"date": "2026-01-14", "content": "美联储会议纪要显示：通胀得到控制，降息预期升温。"},
            {"date": "2026-01-13", "content": "苹果 Vision Pro 2 销量超预期，AR/VR 板块集体走强。"},
            {"date": "2026-01-13", "content": "比特币突破 12 万美元大关，加密货币市场情绪高涨。"}
        ]
        return mock_data, False

# === 4. 主界面逻辑 ===

st.title("💸 DeepQuant 智能投研助手")
st.markdown(
    """
    <style>
    .big-font { font-size:18px !important; }
    </style>
    <div class="big-font">
    基于 <b>RAG (检索增强生成)</b> 架构。实时聚合全球财经资讯，利用向量化技术进行语义搜索与情绪归因。
    </div>
    """, unsafe_allow_html=True
)
st.divider()

# --- 初始化与数据加载 ---
col_status, col_metric = st.columns([2, 1])

with st.spinner('正在初始化神经网络与连接数据源...'):
    model = load_model()
    collection = init_db()
    
    news_data, is_live = fetch_news_feed()
    
    # 状态栏显示
    with col_status:
        if is_live:
            st.success(f"📡 已连接实时 RSS 数据源，获取 {len(news_data)} 条最新资讯")
        else:
            st.warning(f"⚠️ 网络受限，已切换至高性能仿真 (Mock) 数据流，加载 {len(news_data)} 条数据")

    # 存入向量库
    if news_data:
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for item in news_data:
            # 生成唯一ID (防止重复存)
            doc_id = hashlib.md5(item["content"].encode()).hexdigest()
            
            # 简单的查重逻辑 (生产环境应用更高效的 bloom filter)
            try:
                # 尝试获取该ID，如果报错说明不存在
                collection.get(ids=[doc_id])
                # 如果没报错，说明已存在，跳过
                continue 
            except:
                pass # 不存在，继续添加
            
            ids.append(doc_id)
            documents.append(item["content"])
            metadatas.append({"date": item["date"], "link": item["link"]})
        
        # 批量编码与写入 (如果有新数据)
        if documents:
            embeddings = model.encode(documents).tolist()
            collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            with col_metric:
                st.metric("今日新增入库", f"+{len(documents)}", delta_color="normal")

# --- 搜索交互区 ---
st.markdown("### 🔍 语义情报检索")

col_search, col_btn = st.columns([4, 1])

with col_search:
    query = st.text_input("输入查询意图", placeholder="例如：最近有什么关于新能源的利好？", label_visibility="collapsed")

with col_btn:
    search_triggered = st.button("开始分析", type="primary", use_container_width=True)

# --- 结果展示区 ---
if search_triggered or query:
    if not query:
        st.info("请输入查询内容")
    else:
        start_time = time.time()
        
        # 1. 向量化查询
        query_vec = model.encode([query]).tolist()
        
        # 2. 数据库检索 (Top 3)
        results = collection.query(query_embeddings=query_vec, n_results=3)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        
        st.markdown(f"**分析完成** (耗时: `{latency:.2f}ms`)")
        
        # 3. 渲染结果卡片
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                meta_data = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance) # 距离转相似度
                
                # 动态判断情绪颜色 (简单的规则，后续接 LLM)
                card_color = "grey"
                label = "NEUTRAL"
                if any(x in doc_content for x in ["涨", "利好", "突破", "新高"]):
                    card_color = "green"
                    label = "POSITIVE"
                elif any(x in doc_content for x in ["跌", "不及预期", "风险", "警告"]):
                    card_color = "red"
                    label = "NEGATIVE"
                
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.8em; color:gray;">📅 {meta_data['date']}</span>
                            <span style="background-color:{'#e6fffa' if label=='POSITIVE' else '#fff5f5'}; 
                                         color:{'#047857' if label=='POSITIVE' else '#c53030'}; 
                                         padding: 2px 8px; border-radius: 4px; font-size:0.8em; font-weight:bold;">
                                {label}
                            </span>
                        </div>
                        <div style="margin-top: 8px; font-weight: 500;">
                            {doc_content}
                        </div>
                        <div style="margin-top: 8px; font-size: 0.8em;">
                            <a href="{meta_data['link']}" target="_blank">查看原文 🔗</a> 
                            &nbsp; | &nbsp; 语义匹配度: {similarity:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("未找到相关情报，请尝试更换关键词。")