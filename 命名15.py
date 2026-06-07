import requests
import pandas as pd
import time
import os

# ---------------------- 1. 配置区 ----------------------
# 💡 建议多准备 1-2 个 Key 放在这里
KEY_POOL = ["5f9fdf5ee4e81af031d8854e66c5cbdb"] 
CURRENT_KEY_INDEX = 0

OUTPUT_FILE = "河南45县_养老与交通_专项数据.csv"
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

# 专项分类配置：使用了更广的编码
# 养老：涵盖养老院、高龄照料中心、社会福利院
# 交通：涵盖公交站、长途车站、火车站、地铁站
POI_TYPES = [
    {"name": "养老", "types": "090700|090701|090702", "keywords": ""},
    {"name": "交通", "types": "150000|150100|150101|150200|150300", "keywords": ""},
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
            
            for page in range(1, 51):  # 最多爬50页
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
        # 如果文件已存在，则追加；不存在则新建
        header = not os.path.exists(OUTPUT_FILE)
        df.to_csv(OUTPUT_FILE, mode='a', index=False, encoding="utf-8-sig", header=header)
        print(f"💾 数据已成功存入: {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_task()