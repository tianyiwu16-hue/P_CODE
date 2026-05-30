import osmnx as ox
import geopandas as gpd
import pandas as pd

# ====================== 1. 河南45个地区名单（已更新） ======================
counties = [
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市",
    "新安县", "栾川县", "淇县", "新乡县", "沁阳市",
    "林州市", "长葛市", "鄢陵县", "义马市", "渑池县",
    "尉氏县", "兰考县", "洛宁县", "宜阳县", "伊川县",
    "温县", "武陟县", "宝丰县", "舞钢市", "范县",
    "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县",
    "叶县", "鲁山县", "滑县", "内黄县", "封丘县",
    "方城县", "镇平县", "社旗县", "宁陵县", "柘城县",
    "夏邑县", "西华县", "商水县", "太康县", "项城市"
]

# 统一加“河南省”后缀，确保地理编码定位准确
queries = [f"{county}, 河南省" for county in counties]

# 配置设置：开启缓存（避免重复下载），增加超时时间（针对大型路网）
ox.settings.use_cache = True
ox.settings.requests_timeout = 180  # 增加到180秒

# 存储结果
results = []

# ====================== 2. 逐个地区计算 ======================
print(f"🚀 开始处理河南 {len(counties)} 个地区的数据...")

for q in queries:
    county_name = q.replace(", 河南省", "")
    try:
        print(f"正在抓取: {county_name}...", end=" ", flush=True)

        # 1. 获取行政边界并计算面积 (使用 UTM 50N 投影确保精度)
        gdf_bound = ox.geocode_to_gdf(q)
        gdf_bound_proj = gdf_bound.to_crs(epsg=32650)
        area_km2 = gdf_bound_proj.area.iloc[0] / 1e6

        # 2. 获取该区域内的驾车路网
        # 这里的 network_type="drive" 包含了高速、国道、省道及主要的城乡道路
        G = ox.graph_from_place(q, network_type="drive", simplify=True)

        # 3. 计算总长度
        gdf_edges = ox.graph_to_gdfs(G, nodes=False)
        gdf_edges = gdf_edges.to_crs(epsg=32650)
        total_length_km = gdf_edges.length.sum() / 1000

        # 4. 计算路网密度
        density = total_length_km / area_km2

        results.append({
            "地区名称": county_name,
            "公路里程(km)": round(total_length_km, 2),
            "行政面积(km²)": round(area_km2, 2),
            "路网密度(km/km²)": round(density, 4)
        })
        print(" ✅ 完成")

    except Exception as e:
        print(f" ❌ 失败 (原因: {str(e)})")
        # 失败时记录空值，保持表格完整性
        results.append({
            "地区名称": county_name,
            "公路里程(km)": "抓取失败",
            "行政面积(km²)": "抓取失败",
            "路网密度(km/km²)": "抓取失败"
        })
        continue

# ====================== 3. 数据保存 ======================
df = pd.DataFrame(results)
output_filename = "河南45县市_OSM路网密度结果.xlsx"
df.to_excel(output_filename, index=False)

print("\n" + "="*40)
print(f"🎉 统计结束！")
print(f"数据已保存至: {output_filename}")
print("="*40)

















