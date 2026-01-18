import akshare as ak
import pandas as pd
from datetime import datetime

# ================= 🚀 独立行情服务 =================
def get_sector_performance():
    """
    获取 A 股行业板块实时涨跌幅
    返回: { '半导体': 2.5, '房地产': -1.2, ... }
    """
    print(f"[{datetime.now().strftime('%H:%M')}] 正在拉取 Akshare 行情...")
    try:
        # 获取东方财富行业板块实时行情
        # 接口文档: https://akshare.akfamily.xyz/data/stock/stock.html#id5
        df = ak.stock_board_industry_name_em()
        
        # 清洗数据：只留 板块名称 和 涨跌幅
        # 注意：不同版本的 akshare 列名可能不同，通常是 "板块名称", "涨跌幅"
        market_map = {}
        for _, row in df.iterrows():
            name = row['板块名称']
            change = row['涨跌幅']
            market_map[name] = change
            
        print(f"✅ 行情获取成功: 覆盖 {len(market_map)} 个板块")
        return market_map
    except Exception as e:
        print(f"❌ 行情获取失败: {e}")
        return {}

if __name__ == "__main__":
    # 测试一下
    data = get_sector_performance()
    print("半导体涨幅:", data.get('半导体', '未找到'))