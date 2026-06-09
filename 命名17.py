import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
API_KEY = "5f9fdf5ee4e81af031d8854e66c5cbdb" # 👈 请填入你的Key
OUTPUT_FILE = "三省135县_行政村点位剩余数据3.csv"
KEYWORDS = "村委会|村民委员会"
DELAY = 0.2  # 频率控制

# 135县名单 (部分展示，建议通过外部Excel读取或补全)
# 格式: { "行政编码": "县名" }
COUNTY_LIST = {
    # 浙江 45 县
    "331081": "温岭市",
    "331082": "临海市", "331022": "三门县", "331023": "天台县", "331024": "仙居县", "331121": "青田县",
    "331122": "缙云县", "331123": "遂昌县", "331125": "云和县", "331126": "庆元县", "331127": "景宁县"
}

# ---------------------- 2. 爬取逻辑 ----------------------

def crawl_villages():
    all_villages = []
    
    for adcode, county_name in COUNTY_LIST.items():
        print(f"📡 正在扫描: {county_name} ({adcode})...")
        
        for page in range(1, 100):  # 高德单次搜索上限约100页
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "key": API_KEY,
                "keywords": KEYWORDS,
                "city": adcode,
                "citylimit": "true",
                "offset": 25,
                "page": page,
                "output": "json"
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if data.get("info") != "OK":
                    print(f"⚠️ {county_name} 接口反馈: {data.get('info')}")
                    break
                
                pois = data.get("pois", [])
                if not pois:
                    break
                
                for p in pois:
                    location = p.get("location", "").split(",")
                    all_villages.append({
                        "县名": county_name,
                        "行政代码": adcode,
                        "村庄名称": p.get("name"),
                        "经度": location[0] if len(location)>0 else "",
                        "纬度": location[1] if len(location)>1 else "",
                        "地址": p.get("address"),
                        "类型": p.get("type")
                    })
                
                if len(pois) < 25: break  # 最后一页
                time.sleep(DELAY)
                
            except Exception as e:
                print(f"❗ 网络错误: {e}")
                break
        
        print(f"✅ {county_name} 抓取完成。")

    # 保存数据
    if all_villages:
        df = pd.DataFrame(all_villages)
        # 简单去重：按名称和经纬度去重，防止边界重叠
        df.drop_duplicates(subset=['村庄名称', '经度', '纬度'], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n🎉 任务全部完成！总计获取 {len(df)} 个行政村点位，存入: {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_villages()
    
    
    
    
    
    
    
    
    
    

    
 
    
    
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ---------------------- 1. 工具函数：计算经纬度距离 ----------------------
def haversine(lon1, lat1, lon2, lat2):
    """计算两点间的公里数"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位公里
    return c * r

# ---------------------- 2. 核心计算逻辑 ----------------------
def calculate_coverage(villages_df, poi_df, radius=5.0):
    """
    计算覆盖度
    villages_df: 行政村DataFrame
    poi_df: 设施POI DataFrame
    radius: 覆盖半径（公里）
    """
    if villages_df.empty or poi_df.empty:
        return 0.0
    
    v_coords = villages_df[['经度', '纬度']].values.astype(float)
    p_coords = poi_df[['经度', '纬度']].values.astype(float)
    
    covered_count = 0
    for v_lon, v_lat in v_coords:
        # 只要找到一个POI在半径内，即视为被覆盖
        is_covered = False
        for p_lon, p_lat in p_coords:
            dist = haversine(v_lon, v_lat, p_lon, p_lat)
            if dist <= radius:
                is_covered = True
                break
        if is_covered:
            covered_count += 1
            
    return (covered_count / len(villages_df)) * 100

# ---------------------- 3. 主程序 ----------------------
def run_analysis():
    # A. 加载数据 (请确保文件名与你爬取的一致)
    df_villages = pd.read_csv("三省135县_行政村点位数据.csv")
    df_medical = pd.read_csv("医疗_POI.csv")   # 需包含 县名, 经度, 纬度
    df_education = pd.read_csv("教育_POI.csv") # 需包含 县名, 经度, 纬度
    df_traffic = pd.read_csv("交通_POI.csv")   # 需包含 县名, 经度, 纬度

    results = []
    
    # 获取所有唯一的县名进行循环
    counties = df_villages['县名'].unique()
    
    for county in counties:
        print(f"正在分析: {county}...")
        
        # 筛选该县的数据
        v_sub = df_villages[df_villages['县名'] == county]
        m_sub = df_medical[df_medical['县名'] == county]
        e_sub = df_education[df_education['县名'] == county]
        t_sub = df_traffic[df_traffic['县名'] == county]
        
        # 计算三个维度的覆盖度 (默认5公里)
        m_rate = calculate_coverage(v_sub, m_sub, radius=5.0)
        e_rate = calculate_coverage(v_sub, e_sub, radius=5.0)
        t_rate = calculate_coverage(v_sub, t_sub, radius=5.0)
        
        results.append({
            "县名": county,
            "行政村总数": len(v_sub),
            "医疗覆盖度(%)": round(m_rate, 2),
            "教育覆盖度(%)": round(e_rate, 2),
            "交通覆盖度(%)": round(t_rate, 2)
        })

    # B. 保存结果
    output_df = pd.DataFrame(results)
    output_df.to_excel("三省135县_公共服务覆盖度分析结果.xlsx", index=False)
    print("\n✅ 分析完成！结果已存入: 三省135县_公共服务覆盖度分析结果.xlsx")

if __name__ == "__main__":
    run_analysis()
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

# ====================== 全局设置 ======================
ox.settings.requests_timeout = 300
ox.settings.log_console = False

# ====================== 1. 读取数据 ======================
# 使用 r"" 原始字符串可以有效避免 Windows 路径中的斜杠报错
try:
    villages = pd.read_excel(r"D:\桌面应用\河南数据.xlsx")
    poi = pd.read_excel(r"D:\桌面应用\河南45县汇总数据.xlsx")
    print("✅ 河南数据读取成功")
except Exception as e:
    print(f"❌ 数据读取失败: {e}")
    exit()

# ====================== 2. 网络分析函数 ======================
def network_analysis_county(county_name, facility_type="医疗"):
    print(f"正在处理：{county_name} ({facility_type})")
    
    # 【核心修改点】搜索范围定位到河南省
    place = f"{county_name}, 河南省, 中国"
    try:
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        G = ox.project_graph(G)
    except Exception as e:
        print(f"  ⚠️ {county_name} 路网下载失败: {e}")
        return None

    # 筛选该县的数据
    county_vill = villages[villages["县名"] == county_name].copy()
    county_fac = poi[(poi["县名"] == county_name) & (poi["类型"] == facility_type)].copy()

    if len(county_vill) == 0 or len(county_fac) == 0:
        print(f"  ⚠️ {county_name} 缺少村点或{facility_type}POI数据")
        return None

    # 坐标系转换与匹配节点
    vill_gdf = gpd.GeoDataFrame(
        county_vill, 
        geometry=gpd.points_from_xy(county_vill["经度"], county_vill["纬度"]), 
        crs="EPSG:4326"
    ).to_crs(G.graph["crs"])
    
    fac_gdf = gpd.GeoDataFrame(
        county_fac, 
        geometry=gpd.points_from_xy(county_fac["经度"], county_fac["纬度"]), 
        crs="EPSG:4326"
    ).to_crs(G.graph["crs"])

    vill_nodes = ox.nearest_nodes(G, vill_gdf.geometry.x, vill_gdf.geometry.y)
    fac_nodes = list(set(ox.nearest_nodes(G, fac_gdf.geometry.x, fac_gdf.geometry.y)))

    # 多源 Dijkstra
    lengths = nx.multi_source_dijkstra_path_length(G, fac_nodes, weight="length")

    results = []
    # 重新索引方便匹配 vill_nodes 列表
    county_vill_reset = county_vill.reset_index(drop=True)
    
    for idx, vill in county_vill_reset.iterrows():
        node_id = vill_nodes[idx] 
        dist = lengths.get(node_id, None) 
        
        results.append({
            "村名": vill.get("村名", "未知"),
            "县名": county_name,
            "类型": facility_type,
            "距离_米": dist,
            "覆盖_3km": 1 if (dist is not None and dist <= 3000) else 0
        })

    return pd.DataFrame(results)

# ====================== 3. 批量运行 ======================
all_results = []
counties = villages["县名"].unique()

for c in counties: 
    for t in ["医疗", "教育"]:
        df_res = network_analysis_county(c, t)
        if df_res is not None:
            all_results.append(df_res)

if not all_results:
    print("❌ 未生成任何结果，请检查县名是否匹配")
    exit()

final_df = pd.concat(all_results, ignore_index=True)

# ====================== 4. 汇总保存 ======================
# 过滤掉不可达（None）的记录计算平均值
valid_df = final_df.dropna(subset=["距离_米"])

county_summary = valid_df.groupby(["县名", "类型"]).agg(
    平均距离_米=("距离_米", "mean"),
).reset_index()

# 计算覆盖度
coverage = final_df.groupby(["县名", "类型"])["覆盖_3km"].mean().reset_index()
coverage.rename(columns={"覆盖_3km": "覆盖度_3km"}, inplace=True)

final_summary = pd.merge(county_summary, coverage, on=["县名", "类型"])

# 保存结果，文件名已修改为河南
final_df.to_excel("河南_村级路网可达性.xlsx", index=False)
final_summary.to_excel("河南_县域路网可达性汇总.xlsx", index=False)

print("\n" + "="*30)
print("✅ 河南数据处理完成！")
print(f"结果已保存至：\n1. 河南_村级路网可达性.xlsx\n2. 河南_县域路网可达性汇总.xlsx")
print("="*30)
    
    
    
    
    
    
    
    
    
    
    
    
 
    
 
    
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

# ====================== 全局设置 ======================
ox.settings.requests_timeout = 300
ox.settings.log_console = False

# ====================== 1. 读取数据 ======================
try:
    villages = pd.read_excel("D:/TJJM/02_POI数据/贵州45县_行政村点位数据.xlsx")
    poi = pd.read_excel("D:/TJJM/02_POI数据/贵州POI全统计.xlsx")
    print("✅ 数据读取成功")
except Exception as e:
    print(f"❌ 数据读取失败: {e}")
    exit()

# ====================== 2. 网络分析函数 ======================
def network_analysis_county(county_name, facility_type="医疗"):
    print(f"正在处理：{county_name} ({facility_type})")
    
    place = f"{county_name}, 贵州省, 中国"
    try:
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        G = ox.project_graph(G)
    except Exception as e:
        print(f"  ⚠️ 路网下载失败: {e}")
        return None

    county_vill = villages[villages["县名"] == county_name].copy()
    county_fac = poi[(poi["县名"] == county_name) & (poi["类型"] == facility_type)].copy()

    if len(county_vill) == 0 or len(county_fac) == 0:
        return None

    # 匹配节点
    vill_gdf = gpd.GeoDataFrame(county_vill, geometry=gpd.points_from_xy(county_vill["经度"], county_vill["纬度"]), crs="EPSG:4326").to_crs(G.graph["crs"])
    fac_gdf = gpd.GeoDataFrame(county_fac, geometry=gpd.points_from_xy(county_fac["经度"], county_fac["纬度"]), crs="EPSG:4326").to_crs(G.graph["crs"])

    vill_nodes = ox.nearest_nodes(G, vill_gdf.geometry.x, vill_gdf.geometry.y)
    fac_nodes = list(set(ox.nearest_nodes(G, fac_gdf.geometry.x, fac_gdf.geometry.y)))

    # 多源 Dijkstra：计算图中所有节点到最近设施点的距离
    # 结果是一个字典: {节点ID: 距离}
    lengths = nx.multi_source_dijkstra_path_length(G, fac_nodes, weight="length")

    results = []
    for i, vill in county_vill.iterrows():
        node_id = vill_nodes[i - county_vill.index[0]] # 获取对应行在vill_nodes中的节点
        dist = lengths.get(node_id, None) # 如果找不到，返回 None
        
        results.append({
            "村名": vill.get("村名", "未知"),
            "县名": county_name,
            "类型": facility_type,
            "距离_米": dist, # 保持数值格式，便于后续数学运算
            "覆盖_3km": 1 if (dist is not None and dist <= 3000) else 0
        })

    return pd.DataFrame(results)

# ====================== 3. 批量运行 ======================
all_results = []
counties = villages["县名"].unique()

for c in counties: 
    for t in ["医疗", "教育"]:
        df_res = network_analysis_county(c, t)
        if df_res is not None:
            all_results.append(df_res)

final_df = pd.concat(all_results, ignore_index=True)

# ====================== 4. 汇总保存 ======================
# 过滤掉 None 值后计算平均距离
valid_df = final_df.dropna(subset=["距离_米"])

county_summary = valid_df.groupby(["县名", "类型"]).agg(
    平均距离_米=("距离_米", "mean"),
).reset_index()

# 计算覆盖度（以原始所有村为基数）
coverage = final_df.groupby(["县名", "类型"])["覆盖_3km"].mean().reset_index()
coverage.rename(columns={"覆盖_3km": "覆盖度_3km"}, inplace=True)

final_summary = pd.merge(county_summary, coverage, on=["县名", "类型"])

# 保存结果
final_df.to_excel("贵州_村级路网可达性.xlsx", index=False)
final_summary.to_excel("贵州_县域路网可达性汇总.xlsx", index=False)

print("\n✅ 完成！结果已保存。")
    
 
    
 
    
 
    
 
    
 
    
 
    
 




 


import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

# ====================== 全局设置 ======================
ox.settings.requests_timeout = 300
ox.settings.log_console = False

# ====================== 1. 读取数据 ======================
try:
    villages = pd.read_excel(r"D:\桌面应用\河南数据.xlsx")
    poi = pd.read_excel(r"D:\桌面应用\河南45县汇总数据.xlsx")
    
    # 清洗列名，防止肉眼看不见的空格
    villages.columns = villages.columns.str.strip()
    poi.columns = poi.columns.str.strip()
    
    print("✅ 河南数据读取成功")
except Exception as e:
    print(f"❌ 数据读取失败: {e}")
    exit()

# ====================== 2. 网络分析函数 ======================
def network_analysis_county(county_name, facility_type="医疗"):
    print(f"正在处理：{county_name} ({facility_type})")
    
    place = f"{county_name}, 河南省, 中国"
    try:
        # 下载路网
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
        G = ox.project_graph(G)
    except Exception as e:
        print(f"  ⚠️ {county_name} 路网下载失败: {e}")
        return None

    # 【修正点】根据你提供的表头，这里使用 "核心分类"
    county_vill = villages[villages["县名"] == county_name].copy()
    county_fac = poi[(poi["县名"] == county_name) & (poi["核心分类"] == facility_type)].copy()

    if len(county_vill) == 0:
        print(f"  ⚠️ {county_name} 缺少村点数据")
        return None
    if len(county_fac) == 0:
        print(f"  ⚠️ {county_name} 缺少类型为 '{facility_type}' 的POI数据")
        return None

    # 转换坐标系并匹配路网节点
    vill_gdf = gpd.GeoDataFrame(
        county_vill, 
        geometry=gpd.points_from_xy(county_vill["经度"], county_vill["纬度"]), 
        crs="EPSG:4326"
    ).to_crs(G.graph["crs"])
    
    fac_gdf = gpd.GeoDataFrame(
        county_fac, 
        geometry=gpd.points_from_xy(county_fac["经度"], county_fac["纬度"]), 
        crs="EPSG:4326"
    ).to_crs(G.graph["crs"])

    vill_nodes = ox.nearest_nodes(G, vill_gdf.geometry.x, vill_gdf.geometry.y)
    fac_nodes = list(set(ox.nearest_nodes(G, fac_gdf.geometry.x, fac_gdf.geometry.y)))

    # 多源 Dijkstra
    lengths = nx.multi_source_dijkstra_path_length(G, fac_nodes, weight="length")

    results = []
    county_vill_reset = county_vill.reset_index(drop=True)
    
    for idx, vill in county_vill_reset.iterrows():
        node_id = vill_nodes[idx] 
        dist = lengths.get(node_id, None) 
        
        results.append({
            "村名": vill.get("村名", "未知"),
            "县名": county_name,
            "核心分类": facility_type,
            "距离_米": dist,
            "覆盖_3km": 1 if (dist is not None and dist <= 3000) else 0
        })

    return pd.DataFrame(results)

# ====================== 3. 批量运行 ======================
all_results = []
counties = villages["县名"].unique()

# 注意：请确保你的“核心分类”列里确实有“医疗”和“教育”这两个词
for c in counties: 
    for t in ["医疗", "教育"]:
        df_res = network_analysis_county(c, t)
        if df_res is not None:
            all_results.append(df_res)

if not all_results:
    print("❌ 未能生成任何计算结果，请核对“核心分类”中的关键词是否正确。")
    exit()

final_df = pd.concat(all_results, ignore_index=True)

# ====================== 4. 汇总保存 ======================
valid_df = final_df.dropna(subset=["距离_米"])

county_summary = valid_df.groupby(["县名", "核心分类"]).agg(
    平均距离_米=("距离_米", "mean"),
).reset_index()

coverage = final_df.groupby(["县名", "核心分类"])["覆盖_3km"].mean().reset_index()
coverage.rename(columns={"覆盖_3km": "覆盖度_3km"}, inplace=True)

final_summary = pd.merge(county_summary, coverage, on=["县名", "核心分类"])

# 保存
final_df.to_excel("河南_村级路网可达性.xlsx", index=False)
final_summary.to_excel("河南_县域路网可达性汇总.xlsx", index=False)

print("\n" + "="*30)
print("✅ 处理完成！已解决 KeyError 问题。")
print("="*30)   
 
    
 
    
 


    

    






import pandas as pd

# 1. 读取已经生成好的汇总数据
file_path = r"D:\桌面应用\河南_县域路网可达性汇总.xlsx"
try:
    df = pd.read_excel(file_path)
    print("✅ 成功读取汇总文件")
except Exception as e:
    print(f"❌ 读取失败，请检查文件是否存在: {e}")
    exit()

# 2. 清洗列名和数据（防止空格干扰）
df.columns = df.columns.str.strip()
df["县名"] = df["县名"].str.strip()
df["核心分类"] = df["核心分类"].str.strip()

# 3. 数据透视：将“医疗”和“教育”从行转为列
# 这样每一行就是一个县，列分别是“医疗”和“教育”的覆盖度
pivot_df = df.pivot_table(index="县名", columns="核心分类", values="覆盖度_3km")

# 4. 计算综合覆盖度：(医疗 + 教育) / 2
# 使用 mean(axis=1) 的好处：如果某个县缺了一项，它不会报错，有两项则自动取平均
pivot_df["综合覆盖度"] = pivot_df[["医疗", "教育"]].mean(axis=1)

# 5. 定义你要求的特定 45 县顺序
custom_order = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", 
    "新乡县", "沁阳市", "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", 
    "兰考县", "洛宁县", "宜阳县", "伊川县", "温县", "武陟县", "宝丰县", "舞钢市", 
    "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县", "叶县", "鲁山县", 
    "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县", 
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 6. 按照自定义顺序重排
# reindex 会根据 custom_order 重新排列行
final_report = pivot_df.reindex(custom_order)

# 7. 格式化：将小数转换为百分比格式（可选，方便阅读）
# 如果需要保持原始小数以便后续计算，可以注释掉下面这两行
# final_report["综合覆盖度"] = final_report["综合覆盖度"].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "无数据")

# 8. 保存最终结果
output_name = "河南_45县综合覆盖度排名表.xlsx"
final_report.to_excel(output_name)

print("-" * 30)
print(f"✅ 处理完成！")
print(f"结果已按指定顺序保存至: {output_name}")
print("-" * 30)
print(final_report.head()) # 显示前几行核对
















 
    
 
    
 
    
 
    
 
    
    
    
    
    