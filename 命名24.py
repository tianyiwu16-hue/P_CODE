import geopandas as gpd

gdf = gpd.read_file("D:\桌面应用\贵州\贵州省各村居界4490.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("output.geojson", driver="GeoJSON")







import pandas as pd

# 读取xlsx
df = pd.read_excel(r"D:\TJJM\02_指标数据\POI数据\典型县教育、医疗POI数据.xlsx")

# 保存为csv（UTF-8，不保留索引）
df.to_csv(r"D:\TJJM\02_指标数据\POI数据\典型县教育、医疗POI数据.csv", index=False, encoding="utf-8-sig")








import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. 环境与参数设置
# ---------------------------
SEARCH_RADIUS = 5000  # 5公里搜索半径
PROJ_CRS = "EPSG:3857" # 投影坐标系，用于准确计算距离（米）

# ---------------------------
# 2. 数据加载
# ---------------------------
# A. 加载威宁县行政村矢量数据
# 注意：请确保 shp 属性表中有一个表示人口的字段，默认为 'pop'。如果没有，请修改下面 run_2sfca 的参数
villages_path = r"D:\桌面应用\贵州\贵州省各村居界4490.shp"
villages = gpd.read_file(villages_path)
villages = villages.to_crs(PROJ_CRS)

# B. 加载 POI 设施数据 (CSV)
poi_path = r"D:\桌面应用\典型县教育、医疗POI数据.csv"
# 增加 encoding='utf-8-sig' 解决中文乱码问题
poi_df = pd.read_csv(poi_path, encoding='utf-8-sig')

# 转换为 GeoDataFrame：使用你数据表中实际的中文列名“经度”和“纬度”
geometry = [Point(xy) for xy in zip(poi_df["经度"], poi_df["纬度"])]
poi_gdf = gpd.GeoDataFrame(poi_df, geometry=geometry)
poi_gdf = poi_gdf.set_crs(epsg=4326).to_crs(PROJ_CRS)

# ---------------------------
# 3. 提取医疗与教育设施 (根据你的实际列名“类型”过滤)
# ---------------------------
medical = poi_gdf[poi_gdf["类型"].str.contains("医院|卫生院|医务室|诊所|社区卫生", na=False)].copy()
education = poi_gdf[poi_gdf["类型"].str.contains("小学|中学|初中|高中|教学点", na=False)].copy()

# 设置供给能力：由于你的表中没有 capacity 列，统一默认设为 1（代表 1 个机构的服务能力）
medical["supply"] = 1
education["supply"] = 1

# ---------------------------
# 4. 核心函数：两步移动搜索法 (2SFCA)
# ---------------------------
def run_2sfca(demand_points, supply_points, radius, demand_field='pop'):
    """
    demand_points: 行政村 (GeoDataFrame)
    supply_points: 医疗/教育设施 (GeoDataFrame)
    radius: 搜索半径 (米)
    demand_field: 村级数据中的人口字段名
    """
    # 检查人口字段是否存在，不存在则默认为 1（按村庄个数计）
    if demand_field not in demand_points.columns:
        print(f"⚠️ 警告: 未在村庄数据中找到人口字段 '{demand_field}'，将按每个村庄人口=1计算")
        demand_points[demand_field] = 1

    # --- 第一步：计算设施的服务能力 Rj ---
    supply_gdf = supply_points.copy()
    supply_gdf['Rj'] = 0.0
    sindex = demand_points.sindex
    
    for idx, row in supply_gdf.iterrows():
        buffer = row.geometry.buffer(radius)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = demand_points.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffer)]
        
        total_demand = precise_matches[demand_field].sum()
        if total_demand > 0:
            supply_gdf.at[idx, 'Rj'] = row['supply'] / total_demand

    # --- 第二步：计算村庄的可达性 Ai ---
    demand_gdf = demand_points.copy()
    demand_gdf['access_score'] = 0.0
    sindex_supply = supply_gdf.sindex
    
    for idx, row in demand_gdf.iterrows():
        buffer = row.geometry.buffer(radius)
        possible_fac_index = list(sindex_supply.intersection(buffer.bounds))
        possible_fac = supply_gdf.iloc[possible_fac_index]
        precise_fac = possible_fac[possible_fac.intersects(buffer)]
        
        demand_gdf.at[idx, 'access_score'] = precise_fac['Rj'].sum()
        
    return demand_gdf

# ---------------------------
# 5. 执行计算与标准化
# ---------------------------
print("正在计算威宁县医疗可达性...")
# 这里假设你的人口字段名为 'pop'，如果叫‘人口’请改为 demand_field='人口'
villages = run_2sfca(villages, medical, SEARCH_RADIUS, demand_field='pop') 
villages.rename(columns={'access_score': 'med_access'}, inplace=True)

print("正在计算威宁县教育可达性...")
villages = run_2sfca(villages, education, SEARCH_RADIUS, demand_field='pop')
villages.rename(columns={'access_score': 'edu_access'}, inplace=True)

# 标准化 (0-1)
for col in ['med_access', 'edu_access']:
    v_min, v_max = villages[col].min(), villages[col].max()
    villages[f'{col}_std'] = (villages[col] - v_min) / (v_max - v_min) if v_max > v_min else 0

# ---------------------------
# 6. 可视化绘制
# ---------------------------
# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 医疗图
villages.plot(ax=axes[0], column="med_access_std", cmap="YlOrRd", legend=True, 
              legend_kwds={'label': "标准化可达性 (0-1)"})
axes[0].set_title("威宁县：行政村级医疗空间可达性", fontsize=14)
axes[0].set_axis_off()

# 教育图
villages.plot(ax=axes[1], column="edu_access_std", cmap="YlGnBu", legend=True,
              legend_kwds={'label': "标准化可达性 (0-1)"})
axes[1].set_title("威宁县：行政村级教育空间可达性", fontsize=14)
axes[1].set_axis_off()

plt.tight_layout()
plt.show()

print("✅ 计算完成！")





print("村庄数据的列名：", villages.columns.tolist())







# 看看抓到的数据里，行政区划代码和名称是什么
print(villages[['XZQDM', 'XZQMC']].head())
















import geopandas as gpd
# 读取文件
all_gdf = gpd.read_file(r"D:\桌面应用\贵州\贵州省各村居界4490.shp")

# 打印出 XJMC 这一列的前 20 个不重复的名字，看看威宁到底叫什么
print("--- XJMC 列中的部分名称示例 ---")
print(all_gdf['XJMC'].unique()[:20])

# 尝试模糊搜索包含“威”的名字
print("\n--- 包含‘威’字的名称有 ---")
print(all_gdf[all_gdf['XJMC'].str.contains("威", na=False)]['XJMC'].unique())










import geopandas as gpd

# 请将路径替换为你实际使用的 SHP 文件路径
shp_path = r"D:\桌面应用\贵州\贵州省各村居界4490.shp"

# 只读取文件头而不读取所有数据，速度极快
gdf_header = gpd.read_file(shp_path, rows=0)

# 打印所有列名
print("--- SHP 文件的字段清单 ---")
for i, col in enumerate(gdf_header.columns):
    print(f"{i+1}. {col}")











































