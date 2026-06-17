import sys
print(sys.executable) # 这会告诉你当前 Spyder 到底在用哪个 Python
import numpy
print(numpy.__version__) # 如果这里打印出 1.26.4，就说明完全匹配了！


import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio

print("NumPy 版本:", np.__version__)
print("环境已准备就绪！")













import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --- 1. 配置路径 ---
# 确保文件放在纯英文路径下，以防乱码或路径解析错误
gpkg_path = r"D:\桌面应用\weining\weining.gpkg"
poi_csv = r"D:\桌面应用\典型县教育、医疗POI数据.csv"

# --- 2. 读取数据 ---
print("正在加载数据...")
# 读取 GPKG
villages = gpd.read_file(gpkg_path)

# 确保列名正确：如果人口列原本叫 .sum，这里会自动改为 pop
if '.sum' in villages.columns:
    villages = villages.rename(columns={'.sum': 'pop'})

# 读取 POI 并转换为点图层
pois_df = pd.read_csv(poi_csv)
geometry = [Point(xy) for xy in zip(pois_df['lon'], pois_df['lat'])]
pois_gdf = gpd.GeoDataFrame(pois_df, geometry=geometry, crs="EPSG:4326")

# 【至关重要】将所有数据统一转换为投影坐标系 (米为单位)
# 贵州地区推荐使用 EPSG:4547
villages = villages.to_crs(epsg=4547)
pois_gdf = pois_gdf.to_crs(epsg=4547)

# --- 3. 2SFCA 核心计算函数 ---
def calculate_2sfca(demand_gdf, supply_gdf, radius):
    # 建立缓冲区
    supply_buffer = supply_gdf.copy()
    supply_buffer['geometry'] = supply_buffer.buffer(radius)
    
    # 第一步：计算供应点的供需比 Rj
    sj = gpd.sjoin(demand_gdf, supply_buffer, predicate='within')
    pop_sum = sj.groupby('index_right')['pop'].sum()
    supply_gdf['Rj'] = 1 / pop_sum
    supply_gdf['Rj'] = supply_gdf['Rj'].replace([float('inf'), None], 0)

    # 第二步：计算需求点的可达性 Ai
    demand_buffer = demand_gdf.copy()
    demand_buffer['geometry'] = demand_buffer.buffer(radius)
    ai_join = gpd.sjoin(supply_gdf, demand_buffer, predicate='within')
    accessibility = ai_join.groupby('index_right')['Rj'].sum()
    return accessibility

# --- 4. 分别计算 ---
# 假设 CSV 中 'type' 列区分 'education' 和 'healthcare'
edu_gdf = pois_gdf[pois_gdf['type'] == 'education']
med_gdf = pois_gdf[pois_gdf['type'] == 'healthcare']

villages['edu_access'] = calculate_2sfca(villages, edu_gdf, 5000)   # 教育5km
villages['med_access'] = calculate_2sfca(villages, med_gdf, 10000)  # 医疗10km

# --- 5. 保存结果 ---
villages.fillna(0, inplace=True)
villages.to_file(r"D:\gis_data\weining_accessibility_result.gpkg", driver="GPKG")

print("✅ 计算完成！结果已保存在: D:\gis_data\weining_accessibility_result.gpkg")
print(villages[['XZQMC', 'pop', 'edu_access', 'med_access']].head())





















