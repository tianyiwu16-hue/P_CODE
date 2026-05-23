import requests
import pandas as pd
import time
import os

# ---------------------- 配置区 ----------------------
AMAP_KEY = "你的高德API Key"  # 替换成你的key
OUTPUT_DIR = "贵州45县POI数据"  # 输出文件夹
PAGE_SIZE = 20  # 每页20条（最大25）
MAX_PAGES = 50  # 单类型最大页数
DELAY = 0.3     # 防封延迟（秒）

# 45个县
COUNTIES = [
    "正安县","道真县","务川县","凤冈县","湄潭县","桐梓县","习水县","余庆县",
    "普定县","镇宁县","关岭县","紫云县",
    "赫章县","纳雍县","大方县","织金县","金沙县","威宁县",
    "江口县","石阡县","德江县","印江县","沿河县","玉屏县","松桃县",
    "黄平县","施秉县","三穗县","天柱县","锦屏县","黎平县","从江县","榕江县","丹寨县","岑巩县","雷山县","台江县","剑河县","镇远县",
    "荔波县","贵定县","瓮安县","独山县","平塘县","罗甸县","长顺县","龙里县",
    "普安县","晴隆县","贞丰县","望谟县","册亨县","安龙县"
]

# 六类POI：名称、类型码、关键词
POI_TYPES = [
    # 教育
    {"name":"小学","types":"141203","keywords":""},
    {"name":"中学","types":"141202","keywords":""},
    # 医疗
    {"name":"卫生院","types":"090102","keywords":""},
    {"name":"诊所","types":"090300","keywords":""},
    {"name":"药店","types":"090601","keywords":""},
    {"name":"疾控中心","types":"090500","keywords":""},
    {"name":"卫生室","types":"","keywords":"卫生室"},
    # 养老
    {"name":"养老院","types":"060000","keywords":"养老院"},
    {"name":"日间照料中心","types":"","keywords":"日间照料中心"},
    # 商业
    {"name":"超市","types":"060101","keywords":""},
    {"name":"农贸市场","types":"060108","keywords":""},
    {"name":"便利店","types":"060102","keywords":""},
    {"name":"农资店","types":"060109","keywords":"农资"},
    {"name":"邮政快递点","types":"070400","keywords":"邮政"},
    # 文体
    {"name":"文化活动中心","types":"140100","keywords":""},
    {"name":"健身场地","types":"080200","keywords":""},
    {"name":"公园","types":"110000","keywords":""},
    {"name":"乡村书屋","types":"140101","keywords":""},
    {"name":"文化广场","types":"110001","keywords":""},
    # 交通
    {"name":"公交站点","types":"150101","keywords":""},
    {"name":"客运站","types":"150102","keywords":""},
]

# ---------------------- 爬取函数 ----------------------
def crawl_county_poi(county, poi_type):
    """单县单类型POI爬取"""
    all_data = []
    url = "https://restapi.amap.com/v3/place/text"
    
    for page in range(1, MAX_PAGES+1):
        params = {
            "key": AMAP_KEY,
            "city": "贵州省",
            "citylimit": "true",    # 严格限定在贵州
            "types": poi_type["types"],
            "keywords": f"{county}{poi_type['keywords']}",
            "offset": PAGE_SIZE,
            "page": page,
            "extensions": "all",     # 返回详细信息
            "output": "json"
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if data.get("status") != "1":
                print(f"【错误】{county}-{poi_type['name']} 第{page}页：{data.get('info')}")
                break
            
            pois = data.get("pois", [])
            if not pois:
                break  # 无数据退出
            
            # 解析需要字段
            for poi in pois:
                lng, lat = poi["location"].split(",") if poi.get("location") else ("","")
                item = {
                    "县名": county,
                    "POI类型": poi_type["name"],
                    "名称": poi.get("name",""),
                    "地址": poi.get("address",""),
                    "经度": lng,
                    "纬度": lat,
                    "电话": poi.get("tel",""),
                    "分类": poi.get("type",""),
                    "省": poi.get("pname",""),
                    "市": poi.get("cityname",""),
                    "区/县": poi.get("adname",""),
                }
                all_data.append(item)
            
            print(f"✅ {county}-{poi_type['name']} 第{page}页：{len(pois)}条")
            time.sleep(DELAY)
            
            # 不足一页则退出
            if len(pois) < PAGE_SIZE:
                break
                
        except Exception as e:
            print(f"❌ 请求异常：{e}")
            time.sleep(2)
            continue
    
    return all_data

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    
    # 遍历县+POI类型
    for county in COUNTIES:
        for poi in POI_TYPES:
            print(f"\n===== 正在爬取：{county} | {poi['name']} =====")
            data = crawl_county_poi(county, poi)
            all_results.extend(data)
            
            # 每类保存临时文件（防中断）
            if data:
                df_temp = pd.DataFrame(data)
                temp_path = f"{OUTPUT_DIR}/{county}_{poi['name']}.csv"
                df_temp.to_csv(temp_path, index=False, encoding="utf-8-sig")
    
    # 合并总表
    if all_results:
        df_total = pd.DataFrame(all_results)
        total_path = f"{OUTPUT_DIR}/贵州45县_六类POI总数据.csv"
        df_total.to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 爬取完成！总记录：{len(df_total)} 条，保存至：{total_path}")
    else:
        print("\n⚠️ 未获取到任何数据")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import os

# 定义相对路径
file_relative_path = r"Henan_POI_Data\Henan_All_Counties_Combined.csv"

# 1. 检查文件是否存在
if os.path.exists(file_relative_path):
    # 2. 获取绝对路径
    abs_path = os.path.abspath(file_relative_path)
    print(f"✅ 找到文件了！")
    print(f"📂 绝对路径是: {abs_path}")
    
    # 3. (可选) 直接打开文件所在的文件夹
    # Windows系统使用 start，Mac使用 open，Linux使用 xdg-open
    os.system(f'start explorer /select,"{abs_path}"') 
else:
    print(f"❌ 在当前目录下未找到文件: {file_relative_path}")
    print("💡 提示：请确认你是否已经成功运行了爬虫脚本。")       
        
        
        
        
        
        
        
        
        
        
###第二    
import requests
import pandas as pd
import time
import os

# ---------------------- 配置区 ----------------------
AMAP_KEY = "dece54de5834553be25fc1c91cfb27f8"  # ⚠️ 请替换成你自己的高德Key
OUTPUT_DIR = "河南45县POI数据"  # 修改输出文件夹名
PAGE_SIZE = 20  
MAX_PAGES = 50  
DELAY = 0.3     

# 河南省 45个县/县级市示例（已替换为河南行政区划名称）
COUNTIES = [
    "中牟县", "巩义市", "荥阳市", "新密市", "新郑市", "登封市", "尉氏县", "兰考县", "杞县", "通许县",
    "汝州市", "舞钢市", "宝丰县", "叶县", "鲁山县", "郏县", "林州市", "安阳县", "汤阴县", "滑县",
    "内黄县", "浚县", "淇县", "辉县市", "卫辉市", "新乡县", "获嘉县", "原阳县", "延津县", "封丘县",
    "沁阳市", "孟州市", "修武县", "博爱县", "武陟县", "温县", "清丰县", "南乐县", "范县", "台前县",
    "濮阳县", "长葛市", "禹州市", "鄢陵县", "襄城县"
]

# 六类POI配置（保持不变）
POI_TYPES = [
    {"name":"小学","types":"141203","keywords":""},
    {"name":"中学","types":"141202","keywords":""},
    {"name":"卫生院","types":"090102","keywords":""},
    {"name":"诊所","types":"090300","keywords":""},
    {"name":"药店","types":"090601","keywords":""},
    {"name":"疾控中心","types":"090500","keywords":""},
    {"name":"卫生室","types":"","keywords":"卫生室"},
    {"name":"养老院","types":"060000","keywords":"养老院"},
    {"name":"日间照料中心","types":"","keywords":"日间照料中心"},
    {"name":"超市","types":"060101","keywords":""},
    {"name":"农贸市场","types":"060108","keywords":""},
    {"name":"便利店","types":"060102","keywords":""},
    {"name":"农资店","types":"060109","keywords":"农资"},
    {"name":"邮政快递点","types":"070400","keywords":"邮政"},
    {"name":"文化活动中心","types":"140100","keywords":""},
    {"name":"健身场地","types":"080200","keywords":""},
    {"name":"公园","types":"110000","keywords":""},
    {"name":"乡村书屋","types":"140101","keywords":""},
    {"name":"文化广场","types":"110001","keywords":""},
    {"name":"公交站点","types":"150101","keywords":""},
    {"name":"客运站","types":"150102","keywords":""},
]

# ---------------------- 爬取函数 ----------------------
def crawl_county_poi(county, poi_type):
    all_data = []
    url = "https://restapi.amap.com/v3/place/text"
    
    for page in range(1, MAX_PAGES+1):
        params = {
            "key": AMAP_KEY,
            "city": "河南省",  # 👈 这里已修改为河南省
            "citylimit": "true",
            "types": poi_type["types"],
            "keywords": f"{county}{poi_type['keywords']}",
            "offset": PAGE_SIZE,
            "page": page,
            "extensions": "all",
            "output": "json"
        }
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if data.get("status") != "1":
                print(f"【错误】{county}-{poi_type['name']} 第{page}页：{data.get('info')}")
                break
            
            pois = data.get("pois", [])
            if not pois:
                break 
            
            for poi in pois:
                lng, lat = poi["location"].split(",") if poi.get("location") else ("","")
                item = {
                    "县名": county,
                    "POI类型": poi_type["name"],
                    "名称": poi.get("name",""),
                    "地址": poi.get("address",""),
                    "经度": lng,
                    "纬度": lat,
                    "电话": poi.get("tel",""),
                    "分类": poi.get("type",""),
                    "省": poi.get("pname",""),
                    "市": poi.get("cityname",""),
                    "区/县": poi.get("adname",""),
                }
                all_data.append(item)
            
            print(f"✅ {county}-{poi_type['name']} 第{page}页：{len(pois)}条")
            time.sleep(DELAY)
            
            if len(pois) < PAGE_SIZE:
                break
                
        except Exception as e:
            print(f"❌ 请求异常：{e}")
            time.sleep(2)
            continue
    
    return all_data

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    
    for county in COUNTIES:
        for poi in POI_TYPES:
            print(f"\n===== 正在爬取：{county} | {poi['name']} =====")
            data = crawl_county_poi(county, poi)
            all_results.extend(data)
            
            if data:
                df_temp = pd.DataFrame(data)
                temp_path = f"{OUTPUT_DIR}/{county}_{poi['name']}.csv"
                df_temp.to_csv(temp_path, index=False, encoding="utf-8-sig")
    
    if all_results:
        df_total = pd.DataFrame(all_results)
        # 👈 修改总文件名
        total_path = f"{OUTPUT_DIR}/河南45县_六类POI总数据.csv"
        df_total.to_csv(total_path, index=False, encoding="utf-8-sig")
        print(f"\n🎉 爬取完成！总记录：{len(df_total)} 条，保存至：{total_path}")
    else:
        print("\n⚠️ 未获取到任何数据")    
        
        
        
        
        
        
        
        
        
        
###第三
import requests
import pandas as pd
import time
import os

# ---------------------- 配置区 ----------------------
AMAP_KEY = "dece54de5834553be25fc1c91cfb27f8"  # 👈 必填：请替换为你的高德Key
OUTPUT_DIR = "河南POI采集结果"
# 最终汇总的文件名
FINAL_EXCEL_FILE = "河南45县_六类POI总汇总表.xlsx"

PAGE_SIZE = 20  
MAX_PAGES = 50  
DELAY = 0.3     

# 45个县/县级市名单
COUNTIES = [
    # 郑州周边 (5)
    "巩义市", "新郑市", "中牟县", "新密市", "荥阳市",
    
    # 洛阳周边 (3)
    "新安县", "栾川县", "洛宁县", "宜阳县", "伊川县",
    
    # 豫北地区 (4)
    "淇县", "新乡县", "沁阳市", "林州市",
    
    # 许昌周边 (3)
    "长葛市", "鄢陵县", "襄城县",
    
    # 三门峡周边 (2)
    "义马市", "渑池县",
    
    # 开封周边 (2)
    "尉氏县", "兰考县",
    
    # 焦作周边 (2)
    "温县", "武陟县",
    
    # 平顶山周边 (3)
    "宝丰县", "舞钢市", "叶县", "鲁山县",
    
    # 濮阳/漯河 (2)
    "范县", "舞阳县",
    
    # 南阳周边 (6)
    "内乡县", "淅川县", "桐柏县", "方城县", "镇平县", "社旗县",
    
    # 安阳/新乡周边 (3)
    "滑县", "内黄县", "封丘县",
    
    # 商丘周边 (3)
    "宁陵县", "柘城县", "夏邑县",
    
    # 周口周边 (4)
    "西华县", "商水县", "太康县", "项城市"
]

# POI分类配置
POI_TYPES = [
    {"name":"小学","types":"141203","keywords":""},
    {"name":"中学","types":"141202","keywords":""},
    {"name":"卫生院","types":"090102","keywords":""},
    {"name":"诊所","types":"090300","keywords":""},
    {"name":"药店","types":"090601","keywords":""},
    {"name":"疾控中心","types":"090500","keywords":""},
    {"name":"卫生室","types":"","keywords":"卫生室"},
    {"name":"养老院","types":"060000","keywords":"养老院"},
    {"name":"日间照料中心","types":"","keywords":"日间照料中心"},
    {"name":"超市","types":"060101","keywords":""},
    {"name":"农贸市场","types":"060108","keywords":""},
    {"name":"便利店","types":"060102","keywords":""},
    {"name":"农资店","types":"060109","keywords":"农资"},
    {"name":"邮政快递点","types":"070400","keywords":"邮政"},
    {"name":"文化活动中心","types":"140100","keywords":""},
    {"name":"健身场地","types":"080200","keywords":""},
    {"name":"公园","types":"110000","keywords":""},
    {"name":"乡村书屋","types":"140101","keywords":""},
    {"name":"文化广场","types":"110001","keywords":""},
    {"name":"公交站点","types":"150101","keywords":""},
    {"name":"客运站","types":"150102","keywords":""},
]

def crawl_county_poi(county, poi_type):
    """抓取核心逻辑"""
    items_list = []
    url = "https://restapi.amap.com/v3/place/text"
    
    for page in range(1, MAX_PAGES+1):
        params = {
            "key": AMAP_KEY,
            "city": "河南省",
            "citylimit": "true",
            "types": poi_type["types"],
            "keywords": f"{county}{poi_type['keywords']}",
            "offset": PAGE_SIZE,
            "page": page,
            "output": "json"
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data.get("status") == "1":
                pois = data.get("pois", [])
                if not pois: break
                for poi in pois:
                    lng, lat = poi["location"].split(",") if poi.get("location") else ("","")
                    items_list.append({
                        "县名": county,
                        "大类": poi_type["name"],
                        "名称": poi.get("name"),
                        "具体类型": poi.get("type"),
                        "地址": poi.get("address"),
                        "经度": lng,
                        "纬度": lat,
                        "电话": poi.get("tel")
                    })
                if len(pois) < PAGE_SIZE: break
                time.sleep(DELAY)
            else:
                break
        except:
            break
    return items_list

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data_accumulator = [] # 👈 关键：创建一个空列表来装载所有数据

    for county in COUNTIES:
        for poi_info in POI_TYPES:
            print(f"正在抓取: {county} - {poi_info['name']}...")
            result = crawl_county_poi(county, poi_info)
            
            if result:
                # 1. 放入总汇总列表
                all_data_accumulator.extend(result)
                
                # 2. 存一个临时小表（可选，防止程序崩溃）
                pd.DataFrame(result).to_csv(f"{OUTPUT_DIR}/{county}_{poi_info['name']}.csv", index=False, encoding="utf-8-sig")

    # ------------------ 最终合并并导出为一个表格 ------------------
    if all_data_accumulator:
        print("\n正在生成最终汇总表...")
        df_final = pd.DataFrame(all_data_accumulator)
        
        # 导出为 Excel 格式（一个文件，一个工作表）
        # 注意：需要安装 openpyxl 库: pip install openpyxl
        df_final.to_excel(FINAL_EXCEL_FILE, index=False, engine='openpyxl')
        
        # 同时导出一个 CSV 备份（更通用）
        df_final.to_csv("河南45县总汇总.csv", index=False, encoding="utf-8-sig")
        
        print(f"🎉 全部完成！")
        print(f"总计抓取数据：{len(df_final)} 条")
        print(f"汇总表已保存至: {os.path.abspath(FINAL_EXCEL_FILE)}")
    else:
        print("未抓取到任何数据，请检查 API Key 是否有效。")    
        
        
        
        