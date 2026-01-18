import requests
import pandas as pd
import schedule
import time
from datetime import datetime, timedelta
import os
import json
import re
import numpy as np
from collections import defaultdict
from openai import OpenAI

# ================= ⚙️ 配置区 =================
DATA_FILE_PATH = r"C:\Users\12398\Desktop\QAQ\8690project\trade_system_test1\rag_engine\news_data.csv"
DEEPSEEK_API_KEY = ""  # 🔴 必填
BASE_URL = "https://api.deepseek.com"
POLLING_INTERVAL = 2
BACKFILL_COUNT = 60
# ================= 🧠 全局状态 =================
SEEN_NEWS_BUFFER = set()
MARKET_CONTEXT_BUFFER = []
MARKET_CONTEXT_MANUAL = []
SECTOR_HISTORY_BUFFER = []

# ================= 🗺️ 产业链分级图谱 (Knowledge Graph) =================
# 这是给 AI 看的“作战地图”，指导它如何精准打标
SECTOR_KNOWLEDGE = """
【一级大类】 -> 【二级细分 (Sub-Sector)】
1. 人工智能(AI) -> [AI硬件(CPO/算力/服务器), AI应用(游戏/传媒/教育/Sora), AI模型/数据]
2. 半导体 -> [半导体设备(光刻机), 半导体材料, 芯片设计, 封测/制造]
3. 新能源 -> [锂电/固态电池, 光伏, 风电, 储能]
4. 汽车产业链 -> [整车, 汽配/自动驾驶, 飞行汽车(低空)]
5. 医药医疗 -> [创新药/CXO, 中药, 医疗器械]
6. 数字经济 -> [数据要素, 信创/国产软件, 算力租赁]
7. 金融/地产 -> [券商, 银行, 房地产, 保险]
"""

# ================= 🗑️ 噪音黑名单 =================
NOISE_KEYWORDS = [
    "特约", "广告", "报名", "峰会", "论坛", "免责声明",
    "点击查看", "风险提示", "加入圈子", "开户", "上修",
    "大宗交易", "融资融券", "龙虎榜", "汇率", "债市"
]


# ================= 🛠️ 工具函数 =================
def clean_json_string(text):
    # 尝试1: 匹配 Markdown 代码块
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: return match.group(1)

    # 尝试2: 暴力查找第一个 '[' 和最后一个 ']'
    # 这能解决 AI 废话多但包含 JSON 的情况
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        return text[start:end + 1]

    # 如果都找不到，原样返回，让 json.loads 去报错并打印原始内容
    return text


def get_dynamic_half_life(dt):
    """动态半衰期：交易时段加速衰减(4h)，休市时段发酵(24h)"""
    is_workday = dt.weekday() < 5
    hour_float = dt.hour + dt.minute / 60.0
    is_trading_time = is_workday and ((9.5 <= hour_float <= 11.5) or (13.0 <= hour_float <= 15.0))
    return 4.0 if is_trading_time else 24.0


def init_memory():
    global SEEN_NEWS_BUFFER, MARKET_CONTEXT_BUFFER
    if os.path.exists(DATA_FILE_PATH):
        try:
            df = pd.read_csv(DATA_FILE_PATH, encoding='utf-8-sig')
            SEEN_NEWS_BUFFER = set(df['content'].tolist())
            print(f"📚 记忆恢复: {len(SEEN_NEWS_BUFFER)} 条")
        except:
            print("⚠️ 历史文件为空，将创建新文件。")


# ================= 📝 战略内参生成器 (V14.0 结构化版) =================
def generate_daily_brief():
    print("\n☀️ 正在生成【DeepQuant 结构化内参 (V14.0)】...")

    if not os.path.exists(DATA_FILE_PATH):
        print("❌ 无数据。")
        return

    try:
        df = pd.read_csv(DATA_FILE_PATH, encoding='utf-8-sig')

        # 1. 基础清洗
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['impact_score'] = pd.to_numeric(df['impact_score'], errors='coerce').fillna(0)
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0)
        df = df.dropna(subset=['date', 'content'])

        # 2. 周末自适应窗口
        now = datetime.now()
        is_monday = now.weekday() == 0
        lookback_hours = 72 if is_monday else 24
        recent_df = df[df['date'] >= (now - timedelta(hours=lookback_hours))].copy()

        if recent_df.empty:
            print(f"💤 窗口内无数据。")
            return

        # 3. 计算衰减分
        recent_df['half_life'] = recent_df['date'].apply(get_dynamic_half_life)
        recent_df['hours_diff'] = (now - recent_df['date']).dt.total_seconds() / 3600.0
        recent_df['decayed_score'] = recent_df['impact_score'] * (
                    0.5 ** (recent_df['hours_diff'] / recent_df['half_life']))
        recent_df['freshness'] = recent_df['decayed_score'] / (recent_df['impact_score'] + 0.01)

        # 4. 双层聚合统计 (Tiered Aggregation)
        # 先按一级板块分组
        level1_stats = []
        unique_sectors = recent_df['sector'].unique()

        for sector in unique_sectors:
            if sector in ["其他", "全局", "nan"] or not isinstance(sector, str): continue

            sec_df = recent_df[recent_df['sector'] == sector]

            # 一级板块强度 (Top 3 均值)
            l1_strength = sec_df['decayed_score'].sort_values(ascending=False).head(3).mean()
            if l1_strength < 4.0: continue  # 过滤弱板块

            # === 二级细分挖掘 (Drill Down) ===
            sub_stats = []
            unique_subs = sec_df['sub_sector'].unique()
            for sub in unique_subs:
                if not isinstance(sub, str) or sub == "通用": continue
                sub_df = sec_df[sec_df['sub_sector'] == sub]
                # 二级强度
                l2_strength = sub_df['decayed_score'].mean()
                l2_sentiment = sub_df['sentiment'].mean()
                sub_stats.append(f"{sub}(强:{l2_strength:.1f}/情绪:{l2_sentiment:.1f})")

            # 如果没有细分，就空着
            sub_str = " | ".join(sub_stats) if sub_stats else "全板块普涨"

            level1_stats.append({
                'sector': sector,
                'strength': round(l1_strength, 2),
                'count': len(sec_df),
                'sub_details': sub_str,
                'top_news': sec_df.sort_values('decayed_score', ascending=False).iloc[0]['summary']
            })

        if not level1_stats: return

        # 排序并生成 Context
        stat_df = pd.DataFrame(level1_stats).sort_values('strength', ascending=False).head(5)
        sector_context = stat_df.to_string(index=False, columns=['sector', 'strength', 'sub_details'])

        # 提取高分新闻详情
        detail_news = recent_df.sort_values('decayed_score', ascending=False).head(12)
        news_text = "\n".join([
                                  f"- [{row['decayed_score']:.1f}分 | {row['sector']}-{row['sub_sector']}] {row['summary']} | 逻辑:{row['logic']}"
                                  for _, row in detail_news.iterrows()])

        # 5. DeepSeek 战略生成 (结构化指令)
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
        prompt = f"""
        你是A股量化基金经理。现在是{now.strftime('%A')}盘前/午间。
        请基于【双层板块结构】分析资金流向。

        【一级板块强弱榜 (Strength)】
        {sector_context}

        【核心情报 (含二级细分)】
        {news_text}

        【策略生成要求】
        1. **结构化主线**: 指出最强的一级板块，并**必须**点出其内部最强的【二级细分】。(例如: "AI板块最强，内部资金正从应用端(游戏)流向硬件端(光模块)")。
        2. **预期差博弈**: 寻找 `freshness` 高(新消息)但尚未体现在 `strength` 上的细分领域。
        3. **避雷指南**: 指出情绪(sentiment)为负的细分领域。
        4. **标的映射**: 必须引用情报中的 `related_stocks`。

        格式：Markdown，分点陈述，拒绝废话。
        """

        response = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3
        )
        print("\n" + "=" * 40 + f"\n📊 DeepQuant 结构化内参\n" + "-" * 40)
        print(response.choices[0].message.content)
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"⚠️ 日报生成失败: {e}")


# ================= 📡 抓取模块 =================
def fetch_cls_news(limit=20):
    timestamp = int(time.time())
    # rn = row number (抓取数量)
    url = f"https://www.cls.cn/nodeapi/telegraphList?rn={limit}&_={timestamp}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.cls.cn/telegraph",
        "Host": "www.cls.cn",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
    }

    try:
        # 增加 timeout 到 15秒，防止网络慢
        resp = requests.get(url, headers=headers, timeout=15)

        # 🚨 调试核心：如果状态码不是 200，打印出来看看
        if resp.status_code != 200:
            print(f"❌ 请求被拒绝! 状态码: {resp.status_code}")
            # print(f"   返回内容: {resp.text[:100]}") # 调试时可解开
            return []

        data = resp.json()

        # 兼容两种返回结构
        items = data.get('data', {}).get('roll_data') or data.get('data', {}).get('telegraph')

        if not items:
            print(f"⚠️ 接口通了但没数据。返回结构可能是变了: {list(data.keys())}")
            return []

        raw_news = []
        for item in items:
            full_text = f"{item.get('title', '')} {item.get('content', '')}".strip()
            if not full_text: continue

            # 财联社时间戳处理
            ctime = item.get('ctime', int(time.time()))
            dt_str = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M')

            raw_news.append({
                "id": str(item.get('id', hash(full_text))),
                "date": dt_str,
                "content": full_text
            })

        return raw_news

    except Exception as e:
        print(f"❌ 网络/解析致命错误: {e}")
        return []

 # ================= 🧠 核心分析 (V14.1 最终定稿版) =================
def analyze_batch(news_list):
    if not news_list: return []

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    # [Context 策略]
    if 'MARKET_CONTEXT_MANUAL' in globals() and MARKET_CONTEXT_MANUAL:
        context_str = MARKET_CONTEXT_MANUAL
    elif MARKET_CONTEXT_BUFFER:
        context_str = "近期热点: " + " | ".join(MARKET_CONTEXT_BUFFER)
    else:
        context_str = "市场情绪中性，等待方向选择"

    batch_input = [{"id": item['id'], "content": item['content']} for item in news_list]

    prompt = f"""
    【背景】市场状态：{context_str}
    【产业链图谱】：{SECTOR_KNOWLEDGE}
    【角色】A股策略分析师。你的任务是穿透噪音，识别【预期差】与【博弈价值】。

    【核心铁律 (按类型匹配)】
    1.  【政策类】：遵循"政策即命令"。
        - **定性**：区分实招(改变资金/规则)与虚招(口号)。
        - **博弈**：必须结合 **{context_str}** 判断。冰点出利好=雪中送炭；高位出利空=降温打击。
    2.  【海外映射】：提及台积电/英伟达/特斯拉/OpenAI等国外巨头的重磅消息时，**必须**关联A股对应产业链及A股对应【二级细分】(如半导体设备/光模块/汽配)，视为高权重指引。
    3.  【个股微观】：
        - **业绩时机**：预告期内增长=明牌(低分)；非预告期突发=预期差(高分)。
        - **合同/订单 (量化标尺)**：
            *   **高能 (7-8分)**：占上年营收比重 **>30%**。
            *   **中性 (5-6分)**：占上年营收比重 **5%-30%**。
            *   **微弱 (0-4分)**：占上年营收比重 **<5%** 或未披露金额。
        - **技术突破**：需明确“获权威认证”或“获量产订单”，否则视为“软信息”打折处理。
        - **资金动作**：注销式回购 > 真金增持 > 承诺不减持 > 口头口号。

    【评分标准 (0-10) - 梯度优化】
    - 9-10分【核弹/结构性颠覆】：极高意外性。如：印花税、限制量化、实控人被抓、非预告期业绩暴雷/暴增等。
    - 7-8分 【高能/强驱动】：实质性利好。如：海外映射爆发、**营收占比>30%大订单**、行业垄断性技术突破等。
    - 6分   【显著/超预期】：明确的利好，且略超市场预期。
    - 4-5分 【关注/明牌】：信息真实但影响微弱/已兑现。如：**营收占比5-30%的中等合同**、预告期内达标预增。
    - 0-3分 【噪音/垃圾】：**营收占比<5%小合同**、纯行情播报、无来源传闻、无关海外事件。

    【输入新闻】
    {json.dumps(batch_input, ensure_ascii=False)}

    【输出JSON列表】
    - `id`: 原样返回
    - `score`: 整数(0-10)
    - `sentiment`: -1.0(空) ~ 1.0(多)。
    - `summary`: 8字内核心标签
    - `sector`: **一级大类** (如: 人工智能, 半导体, 汽车产业链)。政策类无特定板块填"全局"。
    - `sub_sector`: **二级细分** (如: AI硬件, 游戏传媒, 半导体设备)。若无细分填"通用"。
    - `type`: Policy/Micro/Industry/Noise
    - `impact_horizon`: Immediate/Short/Medium
    - `key_trigger`: 政策/业绩/合同/减持/回购/映射/其他
    - `related_stocks`: ["公司名"]
    - `logic`: 【关键】一句犀利点评。
       - **合同类**：必须注明"营收占比约xx%"，以此作为评分依据。
       - **政策类**：点明具体受影响的细分领域 (如"数据要素入表，利好数字经济")。
       - **噪音类**：(0-3分) 直接注明"无增量信息"。
    """
    raw_content = "（未获取到内容）"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=4000
        )
        raw_content = response.choices[0].message.content

        # 清洗
        cleaned_content = clean_json_string(raw_content)

        # 解析
        parsed_data = json.loads(cleaned_content)

        if isinstance(parsed_data, dict):
            parsed_data = [parsed_data]

        return parsed_data


    except json.JSONDecodeError:

        print("\n❌ JSON 解析失败！DeepSeek 返回了非 JSON 内容。")

        print("🔍 案发现场 (Raw Content):")

        print("-" * 20)

        print(raw_content)  # <--- 这行会告诉你真相

        print("-" * 20)

        return []

    except Exception as e:

        print(f"⚠️ AI 调用其他报错: {e}")

        return []


# ================= 🚨 板块共振雷达 (核心升级) =================
def check_sector_resonance(new_items):
    """
    维护 1 小时的滑动窗口，检测【二级细分】的资金共振
    """
    global SECTOR_HISTORY_BUFFER
    now = time.time()

    # 1. 将新数据加入历史缓存 (只存有效数据)
    for item in new_items:
        if item.get('sub_sector') and item['sub_sector'] != '通用':
            SECTOR_HISTORY_BUFFER.append({
                'time': now,
                'sector': item['sector'],
                'sub_sector': item['sub_sector'],
                'score': item.get('score', 0),
                'summary': item['summary']
            })

    # 2. 清理超过 1 小时的数据 (滑动窗口)
    SECTOR_HISTORY_BUFFER = [x for x in SECTOR_HISTORY_BUFFER if now - x['time'] < 3600]

    # 3. 统计数据 (按 二级细分 聚合)
    # 结构: {'光模块': {'total': 3, 'high': 2, 'parent': 'AI'}}
    stats = defaultdict(lambda: {'total': 0, 'high_score': 0, 'parent': '', 'titles': []})

    for x in SECTOR_HISTORY_BUFFER:
        sub = x['sub_sector']
        stats[sub]['total'] += 1
        stats[sub]['parent'] = x['sector']
        if x['score'] >= 7:
            stats[sub]['high_score'] += 1
        stats[sub]['titles'].append(x['summary'])

    # 4. 触发警报
    # 规则: 1小时内，该细分领域新闻数 >=2 且 至少有1条是高能新闻
    # (细分领域新闻少，阈值比一级板块要低一点，灵敏度要高)
    for sub, data in stats.items():
        if data['total'] >= 2 and data['high_score'] >= 1:
            print(f"\n🚨🚨 【资金共振警报】 >>> {data['parent']} - {sub} <<<")
            print(f"   🔥 1小时内爆发 {data['total']} 条消息 (高能: {data['high_score']})")
            print(f"   📝 线索: {' | '.join(list(set(data['titles'])))}")
            print("-" * 30)


# ================= 🚀 主流程 (修复静默假死版) =================
def run_pipeline(is_first_run=False):
    global SEEN_NEWS_BUFFER

    # 1. 抓取
    fetch_limit = 100 if is_first_run else 20
    if is_first_run: print(f"🚀 系统冷启动：回溯历史数据 (Top {fetch_limit})...")

    raw = fetch_cls_news(limit=fetch_limit)
    if not raw:
        if not is_first_run: print(f"[{datetime.now().strftime('%H:%M')}] ⚠️ 源头无数据")
        return

    # 2. 增量筛选
    batch = []
    skipped_count = 0
    for item in raw:
        if item['content'] in SEEN_NEWS_BUFFER:
            skipped_count += 1
            continue
        if any(n in item['content'] for n in NOISE_KEYWORDS): continue
        if len(item['content']) < 8: continue
        batch.append(item)

    # 状态打印
    timestamp = datetime.now().strftime('%H:%M')
    if is_first_run:
        print(f"[{timestamp}] 回溯结束 | 抓取:{len(raw)} | 已存旧闻:{skipped_count} | 新增待分析:{len(batch)}")
    elif not batch:
        return
    else:
        print(f"[{timestamp}] 🔍 发现 {len(batch)} 条新线索，准备分批分析...")

    # 3. 分批 AI 分析 (Chunking) - 核心修复点
    # 每次只喂 5 条，防止 Token 爆炸导致 JSON 截断
    CHUNK_SIZE = 5
    final_data = []

    for i in range(0, len(batch), CHUNK_SIZE):
        chunk = batch[i: i + CHUNK_SIZE]
        print(f"   ☕ 正在分析第 {i + 1}-{min(i + CHUNK_SIZE, len(batch))} 条...")

        # 调用 AI
        results = analyze_batch(chunk)

        # 建立映射
        result_map = {str(res['id']): res for res in results}

        for item in chunk:
            SEEN_NEWS_BUFFER.add(item['content'])
            res = result_map.get(item['id'])

            if res:
                score = res.get('score', 0)
                # 过滤噪音 (0-4分)
                if score > 4:
                    item.update(res)
                    final_data.append(item)
                    print(
                        f"      ✅ [{score}分 | {res.get('sector', '?')}-{res.get('sub_sector', '?')}] {res.get('summary', '')}")
                else:
                    print(f"      🗑️ [噪音] {res.get('summary', '无价值')}")
            else:
                # 如果 AI 返回的列表里没这个 ID，说明分析漏了或者出错
                print(f"      ⚠️ 分析遗漏: {item['content'][:10]}...")

        # 批次间稍微歇一下，防止 API QPS 限制
        time.sleep(1)

    # 4. 内存维护
    if len(SEEN_NEWS_BUFFER) > 2000:
        SEEN_NEWS_BUFFER = set(list(SEEN_NEWS_BUFFER)[-2000:])

    # 5. 后处理与存储
    if final_data:
        check_sector_resonance(final_data)

        df_new = pd.DataFrame(final_data)
        file_exists = os.path.exists(DATA_FILE_PATH) and os.path.getsize(DATA_FILE_PATH) > 0

        try:
            df_new.to_csv(DATA_FILE_PATH, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
            print(f"   💾 本轮入库 {len(final_data)} 条情报")
        except:
            print("   ❌ 写入失败，请关闭 Excel")



if __name__ == "__main__":
    if "sk-" not in DEEPSEEK_API_KEY:
        print("❌ 错误：请先填入 DeepSeek API Key")
    else:
        print(f"\n📡 DeepQuant V14.1 (结构化资金流版) 启动...")
        print(f"🎯 监控频率: {POLLING_INTERVAL} 分钟/轮")

        # 1. 恢复记忆
        init_memory()

        # 2. 立即跑一次
        run_pipeline()

        # 3. 设定定时任务
        schedule.every(POLLING_INTERVAL).minutes.do(run_pipeline)

        # 设定盘前/午间内参生成
        schedule.every().day.at("08:30").do(generate_daily_brief)
        schedule.every().day.at("12:00").do(generate_daily_brief)

        # 4. 守护进程
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑以此停止服务")
                break
            except Exception as e:
                print(f"\n❌ 主循环异常: {e} (5秒后重试)")
                time.sleep(5)