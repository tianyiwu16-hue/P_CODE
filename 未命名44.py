import requests
import json

# 请求URL
url = 'https://guba.eastmoney.com/api/getData?path=data/api/Data/GetIndexData'


headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://guba.eastmoney.com/'
}


payload = {
    "param": 'day=1',
    'plat':'Wed',
    "path": "data/api/Data/GetIndexData", 
    "env": 2,
    "origin": "",
    "version": "2022",
    "product": "Guba"
}











import requests
url = "https://guba.eastmoney.com/api/getData?path=data/api/Data/GetIndexData"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://guba.eastmoney.com/",
    "Origin": "https://guba.eastmoney.com",
    "Connection": "keep-alive",
}

cookie_str = (
    "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; "
    "nid_create_time=1762410811391; gvi=feRT-qrGn2zkzSvVr-lYCef07; gvi_create_time=1762410811391; "
    "qgqp_b_id=784e4fb0f790a7016edd65e07c75bb2c; st_si=91124984903058; websitepoptg_api_time=1763271036789; "
    "fullscreengg=1; fullscreengg2=1; st_asi=delete; st_pvi=65571146956567; "
    "st_sp=2025-11-06%2014%3A33%3A31; st_inirUrl=https%3A%2F%2Fchatgpt.com%2F; "
    "st_sn=11; st_psi=20251116145453330-111000300841-0542890641"
)
cookies = dict(item.split("=", 1) for item in cookie_str.split("; "))

payload = {
    "param": "day=1",
    "plat": "Web",
    "path": "data/api/Data/GetIndexData",
    "env": 2,
    "origin": "",
    "version": "2022",
    "product": "Guba",
}

response = requests.post(url, headers=headers, cookies=cookies, data=payload)

if response.status_code == 200:
        data = response.json()
        print(data)
        # 逐行输出字典内容
        if isinstance(data, dict):
            for key, value in data.items():
                # 如果 value 是字典或列表，继续递归处理
                if isinstance(value, (dict, list)):
                    print(f"{key}:")
                    print(json.dumps(value, indent=2))  # 美化输出 JSON
                else:
                    print(f"{key}: {value}")
        
        # 逐行输出列表内容
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                print(f"Item {idx+1}:")
                if isinstance(item, (dict, list)):
                    print(json.dumps(item, indent=2))  # 美化输出 JSON
                else:
                    print(item)
        
        else:
            print(data)  # 输出其他类型数据
   






import requests

# 1. 正确的 URL（不带 path）
url = "https://guba.eastmoney.com/api/getData"

# 2. 完整 Headers（补充 Content-Type）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://guba.eastmoney.com/list,600519.html",
    "Origin": "https://guba.eastmoney.com",
    "Content-Type": "application/json;charset=UTF-8",
    "Connection": "keep-alive",
}

# 3. Cookie（保持不变）
cookie_str = (
    "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; "
    "nid_create_time=1762410811391; gvi=feRT-qrGn2zkzSvVr-lYCef07; gvi_create_time=1762410811391; "
    "qgqp_b_id=784e4fb0f790a7016edd65e07c75bb2c; st_si=91124984903058; websitepoptg_api_time=1763271036789; "
    "fullscreengg=1; fullscreengg2=1; st_asi=delete; st_pvi=65571146956567; "
    "st_sp=2025-11-06%2014%3A33%3A31; st_inirUrl=https%3A%2F%2Fchatgpt.com%2F; "
    "st_sn=11; st_psi=20251116145453330-111000300841-0542890641"
)
cookies = dict(item.split("=", 1) for item in cookie_str.split("; "))

# 4. 修正后的 payload（关键！）
payload = {
    "path": "data/api/Data/GetIndexData",
    "param": {"day": 1},        # ← 字典，不是字符串
    "plat": "Web",              # ← 拼写修正：Web，不是 Wed
    "env": 2,
    "origin": "",
    "version": "2022",
    "product": "Guba"
}

# 5. 发送 JSON 请求
response = requests.post(url, headers=headers, cookies=cookies, json=payload)

# 6. 输出结果
print("Status Code:", response.status_code)
print("Response Text:", response.text[:500])  # 先看原始内容

if response.status_code == 200:
    try:
        data = response.json()
        print("\n✅ 解析成功:")
        print(data)
    except Exception as e:
        print("❌ JSON 解析失败:", e)


















import requests

# === 1. URL ===
url = "https://guba.eastmoney.com/api/getData?path=data/api/Data/GetIndexData"

# === 2. 创建 Session ===
session = requests.Session()

# === 3. 设置 Headers ===
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://guba.eastmoney.com/",
    "Origin": "https://guba.eastmoney.com",
    "Connection": "keep-alive",
})

# === 4. 设置 Cookies ===
cookie_str = (
    "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; "
    "nid_create_time=1762410811391; gvi=feRT-qrGn2zkzSvVr-lYCef07; gvi_create_time=1762410811391; "
    "qgqp_b_id=784e4fb0f790a7016edd65e07c75bb2c; st_si=91124984903058; websitepoptg_api_time=1763271036789; "
    "fullscreengg=1; fullscreengg2=1; st_asi=delete; st_pvi=65571146956567; "
    "st_sp=2025-11-06%2014%3A33%3A31; st_inirUrl=https%3A%2F%2Fchatgpt.com%2F; "
    "st_sn=11; st_psi=20251116145453330-111000300841-0542890641"
)
cookies = dict(item.split("=", 1) for item in cookie_str.split("; "))
session.cookies.update(cookies)  # 使用 session 管理 cookies

# === 5. Payload ===
payload = {
    "param": "day=1",
    "plat": "Wed",
    "path": "data/api/Data/GetIndexData",
    "env": 2,
    "origin": "",
    "version": "2022",
    "product": "Guba",
}

# === 6. 发送请求 ===
response = session.post(url, data=payload)

# === 7. 检查并输出结果 ===
if response.status_code == 200:
    try:
        data = response.json()
        print("✅ 数据获取成功！")
        print(data)
    except Exception as e:
        print("⚠️ 无法解析 JSON，原始响应：")
        print(response.text)
else:
    print(f"❌ 请求失败：{response.status_code}")














