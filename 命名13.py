import requests
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
from pyproj import Transformer
import time

# ================= 配置区域 =================
AMAP_API_KEY = '5f9fdf5ee4e81af031d8854e66c5cbdb' 

# 原始县名列表
counties_raw = ["巩义市", "新郑市", "中牟县", "新密市", "荥阳市", "新安县", "栾川县", "淇县", 
    "新乡县", "沁阳市", "林州市", "长葛市", "鄢陵县", "义马市", "渑池县", "尉氏县", 
    "兰考县", "洛宁县", "宜阳县", "伊川县", "温县", "武陟县", "宝丰县", "舞钢市", 
    "范县", "襄城县", "舞阳县", "内乡县", "淅川县", "桐柏县", "叶县", "鲁山县", 
    "滑县", "内黄县", "封丘县", "方城县", "镇平县", "社旗县", "宁陵县", "柘城县", 
    "夏邑县", "西华县", "商水县", "太康县", "项城市"]

# 初始化投影转换器
transformer = Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)

def get_area_with_fallback(county_name):
    """
    尝试两种搜索方式：1. 河南省+县名  2. 直接搜索县名
    """
    # 策略1：带省份搜索（更精确）
    res = fetch_from_amap(f"河南省{county_name}")
    
    # 如果策略1没拿到数据，尝试策略2：直接搜县名
    if isinstance(res, str) and "未找到" in res:
        res = fetch_from_amap(county_name)
        
    return res

def fetch_from_amap(keyword):
    url = "https://restapi.amap.com/v3/config/district"
    params = {
        'key': AMAP_API_KEY,
        'keywords': keyword,
        'subdistrict': 0,
        'extensions': 'all'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # 调试逻辑
        if data.get('status') != '1':
            return f"API返回错误: {data.get('info')}"
        
        districts = data.get('districts', [])
        if not districts:
            return f"未找到该地区: {keyword}"
        
        # 找到匹配度最高的（通常是第一个）
        target = districts[0]
        polyline = target.get('polyline')
        
        if not polyline:
            return "API未返回边界坐标(polyline为空)"
            
        return calculate_polygon_area(polyline)
        
    except Exception as e:
        return f"请求异常: {str(e)}"

def calculate_polygon_area(polyline_str):
    try:
        rings = polyline_str.split('|')
        polygons = []
        for ring in rings:
            path = ring.split(';')
            coords = [tuple(map(float, p.split(','))) for p in path if ',' in p]
            if len(coords) >= 3:
                if coords[0] != coords[-1]: coords.append(coords[0])
                polygons.append(Polygon(coords))
        
        if not polygons: return 0
        geom = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
        projected_geom = transform(transformer.transform, geom)
        return round(projected_geom.area / 1_000_000, 2)
    except:
        return "解析面积失败"

# ================= 执行区域 =================
results = []
print("--- 启动增强版抓取任务 ---")

for name in counties_raw:
    print(f"正在处理: {name}...", end="", flush=True)
    area = get_area_with_fallback(name)
    results.append({"县名": name, "面积_km2": area})
    print(f" -> 结果: {area}")
    time.sleep(0.5)

# 保存
df = pd.DataFrame(results)
df.to_excel("河南县域面积_增强版1.xlsx", index=False)
print("\n任务结束，请查看生成的 Excel 文件。")




















