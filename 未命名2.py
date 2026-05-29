import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

# ---------------------- 配置区 ----------------------
# 请确保此文件路径与你爬虫生成的文件名一致
INPUT_FILE = r"D:\桌面应用\河南45县汇总数据.csv"
OUTPUT_ANALYSIS_FILE = "河南45县_POI指标分析结果.xlsx"

# 45个县/县级市的面积数据（单位：km²）
# 注意：这些是参考面积，建议根据当地最新的统计年鉴进行核对以确保论文/报告的准确性
COUNTY_AREAS = {
    "巩义市": 1041, "新郑市": 873, "中牟县": 1397, "新密市": 1001, "荥阳市": 908,
    "新安县": 1160, "栾川县": 2477, "淇县": 567, "新乡县": 365, "沁阳市": 623,
    "林州市": 2046, "长葛市": 650, "鄢陵县": 871, "义马市": 112, "渑池县": 1422,
    "尉氏县": 1307, "兰考县": 1116, "洛宁县": 2306, "宜阳县": 1616, "伊川县": 1135,
    "温县": 468, "武陟县": 805, "宝丰县": 722, "舞钢市": 640, "范县": 617,
    "襄城县": 920, "舞阳县": 777, "内乡县": 2465, "淅川县": 2793, "桐柏县": 2397,
    "叶县": 1387, "鲁山县": 2432, "滑县": 1814, "内黄县": 1161, "封丘县": 1220,
    "方城县": 2542, "镇平县": 1580, "社旗县": 1203, "宁陵县": 798, "柘城县": 1048,
    "夏邑县": 1481, "西华县": 1194, "商水县": 1313, "太康县": 1759, "项城市": 1083
}

# 6大指标与爬虫中POI类型的映射关系
CATEGORY_MAP = {
    "教育": ["小学", "中学"],
    "医疗": ["卫生院", "诊所", "药店", "疾控中心", "卫生室"],
    "养老": ["养老院", "日间照料中心"],
    "商业": ["超市", "农贸市场", "便利店", "农资店", "邮政快递点"],
    "文体": ["文化活动中心", "健身场地", "公园", "乡村书屋", "文化广场"],
    "交通": ["公交站点", "客运站"]
}

# ---------------------- 计算核心函数 ----------------------

def calculate_spatial_gini(lngs, lats):
    """
    通过 20x20 网格划分计算空间基尼系数
    公式参考：G = (sum |xi - xj|) / (2 * n^2 * mean)
    """
    if len(lngs) < 10: return 0
    # 将县域划分为 20x20 的网格
    counts, _, _ = np.histogram2d(lngs, lats, bins=20)
    data = counts.flatten()
    data = np.sort(data[data > 0]) # 仅计算有分布的区域或全域
    if len(data) < 2: return 0
    
    n = len(data)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))
    return gini

def get_kde_peak(lngs, lats):
    """
    计算高斯核密度估计的峰值
    """
    if len(lngs) < 5: return 0
    try:
        values = np.vstack([lngs, lats])
        kernel = gaussian_kde(values)
        # 采样数据点上的密度
        densities = kernel(values)
        return np.max(densities)
    except:
        return 0

# ---------------------- 主处理流程 ----------------------

def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件：{INPUT_FILE}，请确认爬虫是否运行成功。")
        return

    print("🚀 正在读取数据并开始分析...")
    df = pd.read_csv(INPUT_FILE)
    
    # 确保经纬度是浮点数
    df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    df = df.dropna(subset=['经度', '纬度'])

    results = []

    for county in COUNTY_AREAS.keys():
        area = COUNTY_AREAS[county]
        # 筛选该县的数据
        df_county = df[df['县名'] == county]
        
        if df_county.empty:
            print(f"⚠️ {county} 没有找到数据，跳过。")
            continue
            
        county_stats = {"县名": county, "土地面积(km2)": area}
        
        # 1. 计算六大分类密度 (单位：个/km2)
        for big_cat, sub_cats in CATEGORY_MAP.items():
            count = df_county[df_county['POI类型'].isin(sub_cats)].shape[0]
            density = count / area
            county_stats[f"{big_cat}POI密度"] = round(density, 4)
        
        # 2. 计算空间基尼系数 (衡量分布均匀度)
        lngs = df_county['经度'].values
        lats = df_county['纬度'].values
        county_stats["POI基尼系数"] = round(calculate_spatial_gini(lngs, lats), 4)
        
        # 3. 计算核密度峰值 (衡量核心区集聚强度)
        county_stats["核密度峰值"] = round(get_kde_peak(lngs, lats), 6)
        
        results.append(county_stats)
        print(f"✅ {county} 分析完成")

    # 汇总输出
    df_output = pd.DataFrame(results)
    
    # 保存结果
    df_output.to_excel(OUTPUT_ANALYSIS_FILE, index=False)
    df_output.to_csv("河南45县_指标分析结果.csv", index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print(f"🎉 分析结果已生成！")
    print(f"文件 1: {OUTPUT_ANALYSIS_FILE} (Excel格式)")
    print(f"文件 2: 河南45县_指标分析结果.csv (CSV备份)")

if __name__ == "__main__":
    run_analysis()       
        
        
        
   
        

    

import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
# 💡 建议多准备 1-2 个 Key 放在这里
KEY_POOL = ["5f9fdf5ee4e81af031d8854e66c5cbdb"] 
CURRENT_KEY_INDEX = 0

OUTPUT_FILE = "河南45县_文体_专项数据.csv"
PAGE_SIZE = 20  
DELAY = 0.4  # 稍微延迟，保护Key

# 45个地区名单
COUNTIES = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 专项分类配置：文体类
# 涵盖：文化活动中心、博物馆、图书馆、体育场馆、健身场所、公园广场
POI_TYPES = [
    {
        "name": "文体", 
        "types": "140100|140300|140500|080100|080200|110101|110102", 
        "keywords": "文化广场|文化活动中心|健身|体育|公园"
    }
]

# ---------------------- 2. 核心引擎 ----------------------

def get_key():
    return KEY_POOL[CURRENT_KEY_INDEX]

def switch_key():
    global CURRENT_KEY_INDEX
    if CURRENT_KEY_INDEX < len(KEY_POOL) - 1:
        CURRENT_KEY_INDEX += 1
        print(f"🔄 切换至备用 Key: {get_key()}")
        return True
    return False

def crawl_task():
    all_data = []
    
    for county in COUNTIES:
        for p_conf in POI_TYPES:
            print(f"📡 正在抓取 [{county}] 的 {p_conf['name']} 数据...")
            
            for page in range(1, 51):  # 最多爬50页（高德限制）
                url = "https://restapi.amap.com/v3/place/text"
                params = {
                    "key": get_key(),
                    "city": county,
                    "citylimit": "true",
                    "types": p_conf["types"],
                    "keywords": p_conf["keywords"],
                    "offset": PAGE_SIZE,
                    "page": page,
                    "output": "json"
                }
                
                try:
                    res = requests.get(url, params=params, timeout=10)
                    json_data = res.json()
                    
                    if json_data.get("info") == "USER_DAILY_QUERY_OVER_LIMIT":
                        print("⚠️ 额度耗尽！")
                        if not switch_key():
                            print("❌ 所有Key均已失效，程序停止。")
                            save_to_csv(all_data)
                            return
                        continue

                    pois = json_data.get("pois", [])
                    if not pois: break
                    
                    for p in pois:
                        loc = p.get("location", "").split(",")
                        all_data.append({
                            "县名": county,
                            "核心分类": p_conf["name"],
                            "名称": p.get("name"),
                            "详细类型": p.get("type"),
                            "经度": loc[0] if len(loc)>0 else "",
                            "纬度": loc[1] if len(loc)>1 else "",
                            "地址": p.get("address")
                        })
                    
                    if len(pois) < PAGE_SIZE: break
                    time.sleep(DELAY)
                    
                except Exception as e:
                    print(f"❗ 出错: {e}")
                    break
                    
    save_to_csv(all_data)

def save_to_csv(data):
    if data:
        df = pd.DataFrame(data)
        header = not os.path.exists(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, encoding="utf-8-sig", header=header)
        print(f"💾 数据已成功存入: {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_task()











import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
KEY_POOL = ["5f9fdf5ee4e81af031d8854e66c5cbdb"] 
CURRENT_KEY_INDEX = 0

OUTPUT_FILE = "河南45县_文体数据_地毯式抓取.csv"
PAGE_SIZE = 20  
DELAY = 0.3  

COUNTIES = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 重点优化：文体类编码扩充
# 080000: 体育休闲服务 (包含所有球场、健身房)
# 140000: 文化教育服务 (包含学校、图书馆、文化馆、展览馆)
# 110200: 公园广场 (农村最主要的文体活动场所)
# 080300: 娱乐场所 (包含影剧院)
POI_TYPES = [
    {"name": "文体", "types": "080000|140000|110200|080300"}
]

# ---------------------- 2. 执行引擎 ----------------------

def get_key(): return KEY_POOL[CURRENT_KEY_INDEX]

def crawl_task():
    all_data = []
    for county in COUNTIES:
        for p_conf in POI_TYPES:
            print(f"📡 正在扫描 [{county}] 的全量文体资源...")
            count_per_county = 0
            
            for page in range(1, 101):  # 深度抓取，增加到100页
                url = "https://restapi.amap.com/v3/place/text"
                params = {
                    "key": get_key(),
                    "city": county,
                    "citylimit": "true",
                    "types": p_conf["types"],
                    "offset": PAGE_SIZE,
                    "page": page,
                    "output": "json"
                }
                
                try:
                    res = requests.get(url, params=params, timeout=10)
                    json_data = res.json()
                    
                    if json_data.get("info") != "OK":
                        print(f"⚠️ API反馈: {json_data.get('info')}")
                        break

                    pois = json_data.get("pois", [])
                    if not pois:
                        if page == 1: print(f"❓ 注意：[{county}] 未搜索到目标点。")
                        break
                    
                    for p in pois:
                        loc = p.get("location", "").split(",")
                        all_data.append({
                            "县名": county,
                            "核心分类": "文体",
                            "名称": p.get("name"),
                            "详细类型": p.get("type"),
                            "经度": loc[0] if len(loc)>0 else "",
                            "纬度": loc[1] if len(loc)>1 else "",
                            "地址": p.get("address")
                        })
                    
                    count_per_county += len(pois)
                    if len(pois) < PAGE_SIZE: break
                    time.sleep(DELAY)
                    
                except Exception as e:
                    print(f"❗ 网络错误: {e}")
                    break
            print(f"✅ [{county}] 抓取完成，共计 {count_per_county} 个点。")
            
    if all_data:
        df = pd.DataFrame(all_data).drop_duplicates(subset=['名称', '经度', '纬度'])
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n💾 总计抓取文体数据 {len(df)} 条，已存入: {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_task()







   





import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

# ---------------------- 1. 配置区 ----------------------
# 💡 请确保此路径指向你刚抓取好的“专项数据”CSV文件
INPUT_FILE = r"D:\桌面应用\河南45县_养老与交通_专项数据.csv"
OUTPUT_FILE = "河南45县_养老与交通_指标计算结果.xlsx"

# 你提供的精确面积数据 (km2)
COUNTY_AREAS = {
    "巩义市": 1026.64, "新郑市": 1072.34, "中牟县": 1434.98, "新密市": 995.7, "荥阳市": 915.31,
    "新安县": 1167.7, "栾川县": 2479.07, "淇县": 570.15, "新乡县": 392.16, "沁阳市": 595.8,
    "林州市": 2060.1, "长葛市": 636.26, "鄢陵县": 868.09, "义马市": 99.59, "渑池县": 1359.92,
    "尉氏县": 1106.3, "兰考县": 1117.45, "洛宁县": 2308.85, "宜阳县": 1620.94, "伊川县": 1050.15,
    "温县": 499.22, "武陟县": 825.44, "宝丰县": 713.58, "舞钢市": 627.95, "范县": 608.48,
    "襄城县": 913.68, "舞阳县": 775.14, "内乡县": 2313.82, "淅川县": 2832.6, "桐柏县": 1913.84,
    "叶县": 1387.79, "鲁山县": 2401.89, "滑县": 1781.51, "内黄县": 1145.23, "封丘县": 1221.39,
    "方城县": 2543.85, "镇平县": 1494.41, "社旗县": 1160.15, "宁陵县": 797.78, "柘城县": 1041.27,
    "夏邑县": 1488.68, "西华县": 1205.78, "商水县": 1267.9, "太康县": 1758.18, "项城市": 1091.32
}

# 仅针对养老和交通
CATEGORIES = ["养老", "交通"]

# ---------------------- 2. 核心数学函数 ----------------------

def calculate_gini(lngs, lats):
    """计算空间基尼系数 (反映资源分布的地理公平性)"""
    if len(lngs) < 3: return 0.0
    # 划分 20x20 空间网格
    counts, _, _ = np.histogram2d(lngs, lats, bins=20)
    data = np.sort(counts.flatten())
    if np.sum(data) == 0: return 0.0
    n = len(data)
    index = np.arange(1, n + 1)
    # 公式: G = (Σ(2i-n-1)xi) / (nΣxi)
    return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

def calculate_kde_peak(lngs, lats):
    """计算核密度峰值 (反映核心区的集聚强度)"""
    if len(lngs) < 3: return 0.0
    try:
        values = np.vstack([lngs, lats])
        kernel = gaussian_kde(values)
        return np.max(kernel(values))
    except:
        return 0.0

# ---------------------- 3. 处理主流程 ----------------------

def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print("🚀 开始读取专项数据...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')

    # 自动匹配列名
    col_name = next((c for c in ["核心分类", "分类", "POI类型"] if c in df.columns), None)
    
    # 基础清洗
    df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    df = df.dropna(subset=['经度', '纬度', col_name])

    final_results = []

    for county, area in COUNTY_AREAS.items():
        print(f"📊 计算中: {county}...")
        # 兼容匹配县名
        c_short = county.replace("市","").replace("县","")
        df_county = df[df['县名'].str.contains(c_short, na=False)]
        
        res = {"县名": county, "精确面积(km2)": area}
        
        for cat in CATEGORIES:
            # 筛选养老或交通
            df_cat = df_county[df_county[col_name].str.contains(cat, na=False)]
            count = len(df_cat)
            lngs, lats = df_cat['经度'].values, df_cat['纬度'].values
            
            # --- 1. POI密度 (个/km2) ---
            res[f"{cat}POI密度"] = round(count / area, 6)
            
            # --- 2. 空间基尼系数 ---
            res[f"{cat}POI基尼系数"] = round(calculate_gini(lngs, lats), 4)
            
            # --- 3. 核密度峰值 ---
            res[f"{cat}POI核密度峰值"] = round(calculate_kde_peak(lngs, lats), 6)
            
        final_results.append(res)

    # 导出结果
    output_df = pd.DataFrame(final_results)
    output_df.to_excel(OUTPUT_FILE, index=False)
    
    print("-" * 30)
    print(f"🎉 专项分析完成！")
    print(f"✅ 生成文件: {OUTPUT_FILE}")
    print(f"💡 提示: 包含45个县的养老与交通全维度指标。")

if __name__ == "__main__":
    run_analysis()
   
    
   
    
   
    
   
import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

# ---------------------- 1. 配置区 ----------------------
# 💡 请确保此路径指向你刚抓取好的文体专项数据 CSV
INPUT_FILE = r"D:\桌面应用\河南45县_文体数据_地毯式抓取.csv"
OUTPUT_FILE = "河南45县_文体指标计算结果.xlsx"

# 精确面积数据 (km2)
COUNTY_AREAS = {
    "巩义市": 1026.64, "新郑市": 1072.34, "中牟县": 1434.98, "新密市": 995.7, "荥阳市": 915.31,
    "新安县": 1167.7, "栾川县": 2479.07, "淇县": 570.15, "新乡县": 392.16, "沁阳市": 595.8,
    "林州市": 2060.1, "长葛市": 636.26, "鄢陵县": 868.09, "义马市": 99.59, "渑池县": 1359.92,
    "尉氏县": 1106.3, "兰考县": 1117.45, "洛宁县": 2308.85, "宜阳县": 1620.94, "伊川县": 1050.15,
    "温县": 499.22, "武陟县": 825.44, "宝丰县": 713.58, "舞钢市": 627.95, "范县": 608.48,
    "襄城县": 913.68, "舞阳县": 775.14, "内乡县": 2313.82, "淅川县": 2832.6, "桐柏县": 1913.84,
    "叶县": 1387.79, "鲁山县": 2401.89, "滑县": 1781.51, "内黄县": 1145.23, "封丘县": 1221.39,
    "方城县": 2543.85, "镇平县": 1494.41, "社旗县": 1160.15, "宁陵县": 797.78, "柘城县": 1041.27,
    "夏邑县": 1488.68, "西华县": 1205.78, "商水县": 1267.9, "太康县": 1758.18, "项城市": 1091.32
}

# ---------------------- 2. 核心数学函数 ----------------------

def calculate_gini(lngs, lats):
    """计算空间基尼系数"""
    if len(lngs) < 5: return 0.0  # 点位太少不具备统计意义
    # 建立网格
    counts, _, _ = np.histogram2d(lngs, lats, bins=20)
    data = np.sort(counts.flatten())
    if np.sum(data) == 0: return 0.0
    n = len(data)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

def calculate_kde_peak(lngs, lats):
    """计算核密度峰值"""
    if len(lngs) < 5: return 0.0
    try:
        values = np.vstack([lngs, lats])
        kernel = gaussian_kde(values)
        return np.max(kernel(values))
    except:
        return 0.0

# ---------------------- 3. 主程序 ----------------------

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    print("🚀 正在加载文体数据...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')

    # 数据预处理
    df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    df = df.dropna(subset=['经度', '纬度'])

    # 识别列名
    col = next((c for c in ["核心分类", "分类", "POI类型"] if c in df.columns), None)

    final_results = []

    for county, area in COUNTY_AREAS.items():
        print(f"📊 正在计算: {county}...")
        # 匹配县名（去除市/县后缀进行模糊匹配）
        c_short = county.replace("市","").replace("县","")
        df_county = df[df['县名'].str.contains(c_short, na=False)]
        
        # 筛选文体类
        df_cat = df_county[df_county[col].str.contains("文体", na=False)] if col else df_county
        
        count = len(df_cat)
        lngs = df_cat['经度'].values
        lats = df_cat['纬度'].values
        
        res = {
            "县名": county,
            "面积(km2)": area,
            "文体POI数量": count,
            "文体POI密度": round(count / area, 6),
            "文体POI基尼系数": round(calculate_gini(lngs, lats), 4),
            "文体POI核密度峰值": round(calculate_kde_peak(lngs, lats), 6)
        }
        final_results.append(res)

    # 导出
    result_df = pd.DataFrame(final_results)
    result_df.to_excel(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"🎉 文体指标计算完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    

   
    
   
  
import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
KEY_POOL = ["5f9fdf5ee4e81af031d8854e66c5cbdb"] 
CURRENT_KEY_INDEX = 0

OUTPUT_FILE = "河南45县_养老院_专项数据_含编号.csv"
PAGE_SIZE = 20  
DELAY = 0.4  

# 45个地区名单及其行政区划代码（已为你配置好编号和Adcode）
COUNTY_MAP = {
    "1": ("巩义市", "410181"), "2": ("新郑市", "410184"), "3": ("中牟县", "410122"),
    "4": ("新密市", "410183"), "5": ("荥阳市", "410182"), "6": ("新安县", "410323"),
    "7": ("栾川县", "410324"), "8": ("淇县", "410622"), "9": ("新乡县", "410721"),
    "10": ("沁阳市", "410882"), "11": ("林州市", "410581"), "12": ("长葛市", "411082"),
    "13": ("鄢陵县", "411024"), "14": ("义马市", "411281"), "15": ("渑池县", "411221"),
    "16": ("尉氏县", "410223"), "17": ("兰考县", "410225"), "18": ("洛宁县", "410328"),
    "19": ("宜阳县", "410327"), "20": ("伊川县", "410329"), "21": ("温县", "410825"),
    "22": ("武陟县", "410823"), "23": ("宝丰县", "410421"), "24": ("舞钢市", "410481"),
    "25": ("范县", "410926"), "26": ("襄城县", "411025"), "27": ("舞阳县", "411121"),
    "28": ("内乡县", "411325"), "29": ("淅川县", "411326"), "30": ("桐柏县", "411330"),
    "31": ("叶县", "410422"), "32": ("鲁山县", "410423"), "33": ("滑县", "410526"),
    "34": ("内黄县", "410527"), "35": ("封丘县", "410727"), "36": ("方城县", "411322"),
    "37": ("镇平县", "411323"), "38": ("社旗县", "411327"), "39": ("宁陵县", "411423"),
    "40": ("柘城县", "411424"), "41": ("夏邑县", "411426"), "42": ("西华县", "411622"),
    "43": ("商水县", "411623"), "44": ("太康县", "411627"), "45": ("项城市", "411681")
}

# 专项分类：养老院 (使用你指定的 080402)
# 提示：如果结果偏少，可尝试改为 "090700|090701|080402"
POI_TYPES = [
    {"name": "养老院", "types": "080402", "keywords": ""}
]

# ---------------------- 2. 核心引擎 ----------------------

def get_key():
    return KEY_POOL[CURRENT_KEY_INDEX]

def switch_key():
    global CURRENT_KEY_INDEX
    if CURRENT_KEY_INDEX < len(KEY_POOL) - 1:
        CURRENT_KEY_INDEX += 1
        print(f"🔄 切换至备用 Key: {get_key()}")
        return True
    return False

def crawl_task():
    all_data = []
    
    for idx, (county_name, adcode) in COUNTY_MAP.items():
        for p_conf in POI_TYPES:
            print(f"📡 [{idx}] 正在抓取 {county_name} 的 {p_conf['name']} 数据...")
            
            for page in range(1, 51):
                url = "https://restapi.amap.com/v3/place/text"
                params = {
                    "key": get_key(),
                    "city": adcode, # 使用 adcode 抓取比县名更精准
                    "citylimit": "true",
                    "types": p_conf["types"],
                    "offset": PAGE_SIZE,
                    "page": page,
                    "output": "json"
                }
                
                try:
                    res = requests.get(url, params=params, timeout=10)
                    json_data = res.json()
                    
                    if json_data.get("info") == "USER_DAILY_QUERY_OVER_LIMIT":
                        print("⚠️ 额度耗尽！")
                        if not switch_key():
                            print("❌ 所有Key均已失效。")
                            save_to_csv(all_data)
                            return
                        continue

                    pois = json_data.get("pois", [])
                    if not pois: break
                    
                    for p in pois:
                        loc = p.get("location", "").split(",")
                        all_data.append({
                            "编号": idx,
                            "县名": county_name,
                            "行政区划代码": adcode,
                            "核心分类": p_conf["name"],
                            "名称": p.get("name"),
                            "详细类型": p.get("type"),
                            "经度": loc[0] if len(loc)>0 else "",
                            "纬度": loc[1] if len(loc)>1 else "",
                            "地址": p.get("address")
                        })
                    
                    if len(pois) < PAGE_SIZE: break
                    time.sleep(DELAY)
                    
                except Exception as e:
                    print(f"❗ 出错: {e}")
                    break
                    
    save_to_csv(all_data)

def save_to_csv(data):
    if data:
        df = pd.DataFrame(data)
        # 存入 CSV
        header = not os.path.exists(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, encoding="utf-8-sig", header=header)
        print(f"💾 数据已存入: {OUTPUT_FILE}，本次新增 {len(data)} 条。")

if __name__ == "__main__":
    crawl_task() 
  
    
  
    
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os

# ---------------------- 1. 配置区 ----------------------
# 填入你刚才爬取的养老院数据文件名
INPUT_FILE = r"D:\桌面应用\河南45县_养老院_专项数据_含编号.csv"
OUTPUT_FILE = "河南45县_养老院_指标计算结果.xlsx"

# 45县面积数据 (单位: km²)
# 若需精确到小数点后两位，可在此处核对数据
AREA_MAP = {
    "巩义市": 1026.64, "新郑市": 1072.34, "中牟县": 1434.98, "新密市": 995.7, "荥阳市": 915.31,
    "新安县": 1167.7, "栾川县": 2479.07, "淇县": 570.15, "新乡县": 392.16, "沁阳市": 595.8,
    "林州市": 2060.1, "长葛市": 636.26, "鄢陵县": 868.09, "义马市": 99.59, "渑池县": 1359.92,
    "尉氏县": 1106.3, "兰考县": 1117.45, "洛宁县": 2308.85, "宜阳县": 1620.94, "伊川县": 1050.15,
    "温县": 499.22, "武陟县": 825.44, "宝丰县": 713.58, "舞钢市": 627.95, "范县": 608.48,
    "襄城县": 913.68, "舞阳县": 775.14, "内乡县": 2313.82, "淅川县": 2832.6, "桐柏县": 1913.84,
    "叶县": 1387.79, "鲁山县": 2401.89, "滑县": 1781.51, "内黄县": 1145.23, "封丘县": 1221.39,
    "方城县": 2543.85, "镇平县": 1494.41, "社旗县": 1160.15, "宁陵县": 797.78, "柘城县": 1041.27,
    "夏邑县": 1488.68, "西华县": 1205.78, "商水县": 1267.9, "太康县": 1758.18, "项城市": 1091.32
}

# ---------------------- 2. 计算函数 ----------------------

def calculate_gini(data_points):
    """计算基尼系数 (越接近0越均衡, 越接近1越不均衡)"""
    if len(data_points) < 3: return 0.0
    arr = np.array(data_points)
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

def calculate_kde_peak(lngs, lats):
    """计算核密度峰值 (反映资源集聚程度)"""
    if len(lngs) < 5: return 0.0
    try:
        values = np.vstack([lngs, lats])
        kernel = gaussian_kde(values)
        return np.max(kernel(values))
    except:
        return 0.0

# ---------------------- 3. 主计算逻辑 ----------------------

def main():
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    results = []

    for county, area in AREA_MAP.items():
        # 筛选该县数据
        subset = df[df['县名'] == county]
        count = len(subset)
        
        # 密度计算
        density = count / area if area > 0 else 0
        
        # 简易基尼系数 (基于点位分布在网格中的数量)
        if count > 0:
            lngs = subset['经度'].astype(float).values
            lats = subset['纬度'].astype(float).values
            
            # 使用简单的网格计数来近似基尼系数
            hist, _, _ = np.histogram2d(lngs, lats, bins=5)
            gini = calculate_gini(hist.flatten())
            kde_peak = calculate_kde_peak(lngs, lats)
        else:
            gini = 0
            kde_peak = 0
            
        results.append({
            "县名": county,
            "养老院数量": count,
            "养老院密度(个/km²)": round(density, 6),
            "养老院分布基尼系数": round(gini, 4),
            "空间集聚峰值(KDE)": round(kde_peak, 4)
        })

    # 输出结果
    pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
    print(f"🎉 计算完成！结果已存入：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    
  
   
    
   
    
   
import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

# ---------------------- 配置区 ----------------------
INPUT_FILE = r"D:\桌面应用\Henan_All_Counties_Combined.csv"
OUTPUT_ANALYSIS_FILE = "河南45县_交通POI分析结果.xlsx"

COUNTY_AREAS = {
    "巩义市": 1026.64, "新郑市": 1072.34, "中牟县": 1434.98, "新密市": 995.7, "荥阳市": 915.31,
    "新安县": 1167.7, "栾川县": 2479.07, "淇县": 570.15, "新乡县": 392.16, "沁阳市": 595.8,
    "林州市": 2060.1, "长葛市": 636.26, "鄢陵县": 868.09, "义马市": 99.59, "渑池县": 1359.92,
    "尉氏县": 1106.3, "兰考县": 1117.45, "洛宁县": 2308.85, "宜阳县": 1620.94, "伊川县": 1050.15,
    "温县": 499.22, "武陟县": 825.44, "宝丰县": 713.58, "舞钢市": 627.95, "范县": 608.48,
    "襄城县": 913.68, "舞阳县": 775.14, "内乡县": 2313.82, "淅川县": 2832.6, "桐柏县": 1913.84,
    "叶县": 1387.79, "鲁山县": 2401.89, "滑县": 1781.51, "内黄县": 1145.23, "封丘县": 1221.39,
    "方城县": 2543.85, "镇平县": 1494.41, "社旗县": 1160.15, "宁陵县": 797.78, "柘城县": 1041.27,
    "夏邑县": 1488.68, "西华县": 1205.78, "商水县": 1267.9, "太康县": 1758.18, "项城市": 1091.32
}

TARGET_CAT = "交通"

# ---------------------- 核心函数 ----------------------

def calculate_gini(lngs, lats):
    if len(lngs) < 10:
        return np.nan
    counts, _, _ = np.histogram2d(lngs, lats, bins=20)
    data = counts.flatten()
    data = np.sort(data)
    n = len(data)
    if np.sum(data) == 0:
        return 0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

def calculate_kde_peak(lngs, lats):
    if len(lngs) < 10:
        return np.nan
    try:
        values = np.vstack([lngs, lats])
        kernel = gaussian_kde(values)
        densities = kernel(values)
        return np.max(densities)
    except:
        return np.nan

# ---------------------- 主程序 ----------------------

def run_analysis():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件：{INPUT_FILE}")
        return

    print("🚀 读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')

    # 自动识别分类列
    col_name = ""
    for name in ["核心分类", "分类", "POI类型"]:
        if name in df.columns:
            col_name = name
            break

    if not col_name:
        print("❌ 找不到分类列")
        return

    print(f"✅ 分类字段：{col_name}")

    # 基础清洗
    df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    df = df.dropna(subset=['经度', '纬度', col_name])

    # ---------------------- ⭐关键优化 ----------------------

    # 1️⃣ 只保留交通
    df = df[df[col_name].str.contains(TARGET_CAT, na=False)]

    # 2️⃣ 名称去括号（去掉进站口/出站口）
    df['名称简化'] = df['名称'].str.replace(r'\(.*?\)', '', regex=True)

    # 3️⃣ 去重（核心）
    df = df.drop_duplicates(subset=['县名', '名称简化'])

    # （可选）4️⃣ 只保留核心交通设施（做论文建议打开）
    # df = df[df['类型编码'].str.contains("火车站|汽车站|机场", na=False)]

    # ------------------------------------------------------

    results = []

    for county, area in COUNTY_AREAS.items():
        print(f"📊 {county}...")

        df_county = df[df['县名'].str.contains(county.replace("县","").replace("市",""))]

        count = len(df_county)

        res = {
            "县名": county,
            "面积_km2": area,
            "交通POI数量": count,
            "交通POI密度": round(count / area, 6)
        }

        lngs = df_county['经度'].values
        lats = df_county['纬度'].values

        res["交通POI基尼系数"] = round(calculate_gini(lngs, lats), 4)
        res["交通POI核密度峰值"] = round(calculate_kde_peak(lngs, lats), 6)

        results.append(res)

    output_df = pd.DataFrame(results)
    output_df.to_excel(OUTPUT_ANALYSIS_FILE, index=False)

    print("🎉 完成！输出文件：", OUTPUT_ANALYSIS_FILE)

# ----------------------
if __name__ == "__main__":
    run_analysis()  
   
    
   
    
   
    
   
    
   
    
        
        
        
        
        
        
        
        
        
        
import requests
import pandas as pd
import time
import os

# ---------------------- 配置区 ----------------------
AMAP_KEY = "5f9fdf5ee4e81af031d8854e66c5cbdb"  # 👈 必须填入你的Key
OUTPUT_DIR = "河南试点5县POI数据"
PAGE_SIZE = 20  
MAX_PAGES = 50  
DELAY = 0.5     # 稍微增加延迟，保证请求稳定性

# 提取前5个地区进行测试
ALL_COUNTIES = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]
COUNTIES = ALL_COUNTIES[:45] # 👈 仅取前5个

# 完善后的POI分类：增加分类码的覆盖范围
POI_TYPES = [
    # 交通类 (针对交通缺失做了重点补全)
    {"name":"交通","types":"150100|150101|150102|150104|150200","keywords":"公交站|火车站|客运站"},
    # 教育
    {"name":"教育","types":"141203|141202|141201","keywords":""},
    # 医疗
    {"name":"医疗","types":"090100|090101|090102|090300|090601|090500","keywords":"卫生院|诊所|药店"},
    # 养老
    {"name":"养老","types":"090700|090701|090702","keywords":"养老院|福利院|老年公寓"},
    # 商业
    {"name":"商业","types":"060100|060101|060102|060108|060109|070400","keywords":"超市|农贸市场|快递"},
    # 文体
    {"name":"文体","types":"140100|080200|110000|140101|110001","keywords":"体育场|图书馆|公园|广场"},
]

# ---------------------- 爬取逻辑 ----------------------
def crawl_poi_refined(county, poi_conf):
    data_list = []
    url = "https://restapi.amap.com/v3/place/text"
    
    for page in range(1, MAX_PAGES + 1):
        params = {
            "key": AMAP_KEY,
            "city": county,          # 👈 直接使用县名作为城市搜索范围
            "citylimit": "true",     # 强制限制在县境内
            "types": poi_conf["types"],
            "keywords": poi_conf["keywords"],
            "offset": PAGE_SIZE,
            "page": page,
            "extensions": "all",     # 获取详细字段
            "output": "json"
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            
            if data.get("status") != "1":
                print(f"  [Error] {county}-{poi_conf['name']}: {data.get('info')}")
                break
            
            pois = data.get("pois", [])
            if not pois:
                break 
            
            for poi in pois:
                # 坐标处理
                loc = poi.get("location", "").split(",")
                lng = loc[0] if len(loc) > 0 else ""
                lat = loc[1] if len(loc) > 1 else ""
                
                # 评分与营业额等深度信息（部分POI可能没有）
                biz_ext = poi.get("biz_ext", {})
                rating = biz_ext.get("rating", "N/A")
                
                data_list.append({
                    "县名": county,
                    "核心分类": poi_conf["name"],
                    "POI名称": poi.get("name"),
                    "POI唯一ID": poi.get("id"),
                    "分类代码": poi.get("typecode"),
                    "全分类描述": poi.get("type"),
                    "详细地址": poi.get("address"),
                    "经度": lng,
                    "纬度": lat,
                    "联系电话": poi.get("tel"),
                    "所属省份": poi.get("pname"),
                    "所属城市": poi.get("cityname"),
                    "所属区县": poi.get("adname"),
                    "评分": rating
                })
            
            print(f"  √ {county} {poi_conf['name']} 第{page}页抓取 {len(pois)} 条")
            time.sleep(DELAY)
            
            if len(pois) < PAGE_SIZE: break
                
        except Exception as e:
            print(f"  [Exception] {e}")
            time.sleep(2)
    return data_list

# ---------------------- 执行 ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_data = []

    print(f"🚀 开始测试前45个地区: {', '.join(COUNTIES)}")

    for county in COUNTIES:
        print(f"\n--- 正在处理: {county} ---")
        for p_type in POI_TYPES:
            results = crawl_poi_refined(county, p_type)
            if results:
                master_data.extend(results)
                # 分量保存
                pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/{county}_{p_type['name']}.csv", index=False, encoding="utf-8-sig")

    if master_data:
        df_final = pd.DataFrame(master_data)
        # 去重：防止不同分类关键词导致抓取到重复的POI
        df_final.drop_duplicates(subset=['POI唯一ID'], inplace=True)
        
        final_path = f"{OUTPUT_DIR}/河南45县汇总数据.csv"
        df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 测试完成！")
        print(f"共抓取不重复记录：{len(df_final)} 条")
        print(f"汇总文件已存至: {final_path}")
    else:
        print("未抓取到数据，请检查 Key 有效期或网络。")     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
KEY_POOL = ["5f9fdf5ee4e81af031d8854e66c5cbdb"] 
CURRENT_KEY_INDEX = 0

OUTPUT_FILE = "河南45县_中学数据_当前快照.csv"
PAGE_SIZE = 20  
DELAY = 0.3  

COUNTIES = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 重点：141202(中学/高中), 141203(初中)
POI_TYPES = [
    {"name": "中学", "types": "141202|141203"}
]

# ---------------------- 2. 执行引擎 ----------------------

def get_key(): return KEY_POOL[CURRENT_KEY_INDEX]

def crawl_schools():
    all_schools = []
    
    for county in COUNTIES:
        print(f"📡 正在获取 [{county}] 的中学列表...")
        county_count = 0
        
        for p_conf in POI_TYPES:
            for page in range(1, 25):  # 学校数量通常不会超过500个，25页足够
                url = "https://restapi.amap.com/v3/place/text"
                params = {
                    "key": get_key(),
                    "city": county,
                    "citylimit": "true",
                    "types": p_conf["types"],
                    "offset": PAGE_SIZE,
                    "page": page,
                    "output": "json"
                }
                
                try:
                    res = requests.get(url, params=params, timeout=10)
                    data = res.json()
                    
                    if data.get("info") != "OK":
                        print(f"⚠️ API错误: {data.get('info')}")
                        break

                    pois = data.get("pois", [])
                    if not pois: break
                    
                    for p in pois:
                        loc = p.get("location", "").split(",")
                        all_schools.append({
                            "县名": county,
                            "学校名称": p.get("name"),
                            "详细类型": p.get("type"),
                            "经度": loc[0] if len(loc)>0 else "",
                            "纬度": loc[1] if len(loc)>1 else "",
                            "地址": p.get("address")
                        })
                    
                    county_count += len(pois)
                    if len(pois) < PAGE_SIZE: break
                    time.sleep(DELAY)
                    
                except Exception as e:
                    print(f"❗ 网络异常: {e}")
                    break
        print(f"✅ [{county}] 找到 {county_count} 所中学")

    # 导出并统计数量
    df = pd.DataFrame(all_schools)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    # 生成统计简表
    summary = df.groupby("县名").size().reset_index(name="2026年中学数量")
    summary.to_excel("河南45县_中学数量统计对照表.xlsx", index=False)
    
    print("\n" + "="*30)
    print(f"💾 详细名单存入: {OUTPUT_FILE}")
    print(f"📊 数量统计表存入: 河南45县_中学数量统计对照表.xlsx")

if __name__ == "__main__":
    crawl_schools()      
        
        
        
        
        
        
        
        


import requests
import pandas as pd
import time


# ====================== 【请在这里填你的高德Web服务Key】 ======================
AMAP_KEY = "5f9fdf5ee4e81af031d8854e66c5cbdb"

# 贵州45县名单（已修正为高德标准名称）
county_list = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 正确的交通POI类型代码（高德官方标准）
# 全量交通POI：150200=火车站, 150400=长途汽车站, 150500=地铁站, 150700=公交车站, 150800=班车站,151100=出租车,151200=轮渡站
# 全量医疗POI：
# 全量教育POI：141202=中学，141203=小学，
traffic_types = "141202"
poi_all = []

# ====================== 开始爬取（修正检索逻辑） ======================
for county in county_list:
    print(f"正在爬取：{county} 的POI...")
    page = 1
    while True:
        url = ( f"https://restapi.amap.com/v3/place/text?"
         f"keywords={county}&types=141202|141203&city=河南省&citylimit=true"
         f"&offset=25&page={page}&key={AMAP_KEY}&extensions=all"
     )
        # ====================== 【河南版 - 修改后的URL构造段】 ======================
# 直接替换你原来的那段 f-string 即可（已适配河南省 + 中学POI类型）

          # ====================== 【河南版 - 修改后的URL构造段】 ======================
# 直接替换你原来的那段 f-string 即可（已适配河南省 + 中学POI类型）

                 
   
        # 【核心修正】直接用「县名+交通类型」精准检索，避免全省数据污染
        # ====================== 【河南版 - 修改后的URL构造段】 ======================
# 直接替换你原来的那段 f-string 即可（已适配河南省 + 中学POI类型）


        try:
            res = requests.get(url, timeout=15)
            data = res.json()

            # 接口状态判断
            if data.get("status") != "1":
                print(f"{county} 第{page}页请求异常：{data.get('info')}")
                break

            pois = data.get("pois", [])
            if not pois:
                print(f"{county} 爬取完成，共{page-1}页")
                break

            # 数据清洗与存储
            for p in pois:
                # 二次校验：确保POI归属当前县，避免跨县数据
                adname = p.get("adname", "")
                if county not in adname:
                    continue

                location = p.get("location", "")
                lng, lat = location.split(",") if location else ("", "")

                poi_all.append({
                    "县名": county,
                    "POI名称": p.get("name", ""),
                    "地址": p.get("address", ""),
                    "经度": lng,
                    "纬度": lat,
                    "类型": p.get("type", ""),
                    "大类": p.get("typecode", "")
                })

            page += 1
            time.sleep(0.5)  # 防限流，避免API报错

        except Exception as e:
            print(f"{county} 爬取出错：{str(e)}")
            break

# ====================== 保存Excel ======================
df = pd.DataFrame(poi_all)
df.to_excel("贵州45县_交通POI_公交站_客运站.xlsx", index=False)
print("\n✅ 爬取完成！文件已保存：贵州45县_交通POI_公交站_客运站_最终.xlsx")
print(f"共获取{len(df)}条交通POI数据")
        






import requests
import pandas as pd
import time
import os

# 配置
AMAP_KEY = "5f9fdf5ee4e81af031d8854e66c5cbdb"
COUNTIES = ["巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
 "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
 "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
 "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
 "夏邑县", "西华县", "商水县", "太康县", "项城市"] # 填入你的45县列表

def fetch_county_schools(county):
    all_data = []
    # 141202=中学, 141203=小学(你可以根据需要调整)
    types = "141202|141203" 
    
    for page in range(1, 50):
        url = f"https://restapi.amap.com/v3/place/text?key={AMAP_KEY}&city={county}&types={types}&offset=25&page={page}&extensions=all&citylimit=true"
        try:
            res = requests.get(url, timeout=10).json()
            if res.get("status") != "1" or not res.get("pois"): break
            
            for p in res["pois"]:
                all_data.append({
                    "县名": county,
                    "名称": p.get("name"),
                    "经度": p.get("location", "").split(",")[0],
                    "纬度": p.get("location", "").split(",")[1]
                })
            time.sleep(0.3)
        except: break
    return all_data

# 执行抓取
final_list = []
for c in COUNTIES:
    final_list.extend(fetch_county_schools(c))

df = pd.DataFrame(final_list)
df.to_csv("河南45县_2026中学数据.csv", index=False, encoding="utf-8-sig")








import requests
import pandas as pd
import time

# 配置区
AMAP_KEY = "5f9fdf5ee4e81af031d8854e66c5cbdb"
COUNTIES = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

def get_school_count():
    stats = []
    
    for county in COUNTIES:
        print(f"📡 正在统计: {county}...")
        total_count = 0
        # 141202=中学, 141203=初中
        types = "141202"
        
        # 为了获取准确总数，查看API返回的 count 字段
        url = f"https://restapi.amap.com/v3/place/text?key={AMAP_KEY}&city={county}&types={types}&offset=1&page=1&extensions=all"
        
        try:
            res = requests.get(url, timeout=10).json()
            if res.get("status") == "1":
                total_count = int(res.get("count", 0))
            else:
                total_count = 0
        except Exception as e:
            print(f"❌ {county} 查询失败: {e}")
            total_count = -1 # 标记错误
            
        stats.append({"县名": county, "中学总数": total_count})
        time.sleep(0.3) # 遵守API速率限制

    # 导出统计表
    df = pd.DataFrame(stats)
    df.to_excel("河南45县_中学数量统计表.xlsx", index=False)
    print("\n✅ 统计完成！已保存为: 河南45县_中学数量统计表.xlsx")
    return df

if __name__ == "__main__":
    df = get_school_count()
    print(df)





# 河南45县行政区划对照表
# 使用字典结构，键为编号，值为(名称, 行政区划代码)
HENAN_COUNTIES = {
    "1": ("巩义市", "410181"), "2": ("新郑市", "410184"), "3": ("中牟县", "410122"),
    "4": ("新密市", "410183"), "5": ("荥阳市", "410182"), "6": ("新安县", "410323"),
    "7": ("栾川县", "410324"), "8": ("淇县", "410622"), "9": ("新乡县", "410721"),
    "10": ("沁阳市", "410882"), "11": ("林州市", "410581"), "12": ("长葛市", "411082"),
    "13": ("鄢陵县", "411024"), "14": ("义马市", "411281"), "15": ("渑池县", "411221"),
    "16": ("尉氏县", "410223"), "17": ("兰考县", "410225"), "18": ("洛宁县", "410328"),
    "19": ("宜阳县", "410327"), "20": ("伊川县", "410329"), "21": ("温县", "410825"),
    "22": ("武陟县", "410823"), "23": ("宝丰县", "410421"), "24": ("舞钢市", "410481"),
    "25": ("范县", "410926"), "26": ("襄城县", "411025"), "27": ("舞阳县", "411121"),
    "28": ("内乡县", "411325"), "29": ("淅川县", "411326"), "30": ("桐柏县", "411330"),
    "31": ("叶县", "410422"), "32": ("鲁山县", "410423"), "33": ("滑县", "410526"),
    "34": ("内黄县", "410527"), "35": ("封丘县", "410727"), "36": ("方城县", "411322"),
    "37": ("镇平县", "411323"), "38": ("社旗县", "411327"), "39": ("宁陵县", "411423"),
    "40": ("柘城县", "411424"), "41": ("夏邑县", "411426"), "42": ("西华县", "411622"),
    "43": ("商水县", "411623"), "44": ("太康县", "411627"), "45": ("项城市", "411681")
}

def generate_county_table():
    """生成带有编号的统计表"""
    import pandas as pd
    data = []
    for num, (name, code) in HENAN_COUNTIES.items():
        data.append({"编号": num, "县名": name, "行政区划代码": code})
    
    df = pd.DataFrame(data)
    # 存为 Excel，方便你后续对照
    df.to_excel("河南45县_行政编码表.xlsx", index=False)
    print("✅ 河南45县行政编码表已生成，请查看：河南45县_行政编码表.xlsx")

if __name__ == "__main__":
    generate_county_table()
























