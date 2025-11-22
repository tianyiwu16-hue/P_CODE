import os
import re
import csv
import time
import zipfile
import requests
from bs4 import BeautifulSoup

student_id = "2024113419"
name = "许志国"
classname = "数据2402"

folder_name = f"{student_id}_{name}_{classname}"
data_folder = os.path.join(folder_name, "data")
csv_file = os.path.join(folder_name, "newslist.csv")
zip_name = f"{folder_name}.zip"

os.makedirs(data_folder, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

base_url = "https://finance.eastmoney.com/yaowen.html"
print("正在抓取财经新闻网页...")

resp = requests.get(base_url, headers=headers, timeout=(5,10))
resp.encoding = "utf-8"
soup = BeautifulSoup(resp.text, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    title = a.get_text(strip=True)
    # 过滤仅保留财经新闻页面
    if re.match(r"^https?://finance\.eastmoney\.com/a/\d{8,}", href):
        links.append((title, href))

print(f"找到 {len(links)} 条潜在新闻链接")

news_data = []
titles_seen = set()

for i, (title, link) in enumerate(links[:30]):  # 限制30条
    if not title or title in titles_seen:
        continue
    titles_seen.add(title)
    try:
        detail = requests.get(link, headers=headers, timeout=(5,10))
        detail.encoding = "utf-8"
        text = BeautifulSoup(detail.text, "html.parser").get_text("\n", strip=True)
    except Exception as e:
        text = f"【正文获取失败：{e}】"

    safe_title = "".join(c for c in title if c not in "\\/:*?\"<>|")[:50]
    txt_path = os.path.join(data_folder, f"{safe_title}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    news_data.append([title, link])
    print(f"✅ 抓取成功：{title}")
    time.sleep(0.5)

# 保存CSV
with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["标题", "链接"])
    writer.writerows(news_data)

# 打包ZIP
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            path = os.path.join(root, file)
            arc = os.path.relpath(path, start=os.path.dirname(folder_name))
            zipf.write(path, arc)

print(f"🎉 共抓取 {len(news_data)} 条新闻，数据已打包：{zip_name}")

print("压缩文件完整路径：", os.path.abspath(zip_name))












import requests
import json

# 定义 API URL
url = "https://guba.eastmoney.com/api/getData?path=data/api/Data/GetIndexData"

# 定义请求头（如果需要的话）
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# 发送 GET 请求
response = requests.get(url, headers=headers)

# 检查请求是否成功
if response.status_code == 200:
    data = response.json()  # 解析返回的 JSON 数据
    
    # 打印数据（你可以根据需要修改这部分代码来保存或处理数据）
    print(json.dumps(data, indent=4, ensure_ascii=False))
else:
    print("获取数据失败，状态码:", response.status_code)














import requests
import json
import time

# === 配置参数 ===
stock_code = "600519"  # 股票代码（必须是纯数字）
page = 1
page_size = 20

url = "https://guba.eastmoney.com/api/getData"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': f'https://guba.eastmoney.com/list,{stock_code}.html',
    'Content-Type': 'application/json;charset=UTF-8',
    'Accept': 'application/json'
}

payload = {
    "path": "data/api/Data/GetIndexData",
    "param": {
        "code": stock_code,
        "sort": "time",        # "time" 最新发帖，"reply" 回复最多
        "page": page,
        "pageSize": page_size
    }
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print("Status Code:", response.status_code)
    
    if response.status_code == 200:
        data = response.json()
        # 打印原始响应（调试用）
        # print("Raw Response:", json.dumps(data, indent=2, ensure_ascii=False))
        
        if data.get("re") and data.get("result"):
            posts = data["result"]
            print(f"\n✅ 成功获取 {len(posts)} 条帖子\n")
            
            for i, post in enumerate(posts, 1):
                title = post.get("title", "无标题")
                author = post.get("author", "匿名")
                create_time = post.get("createTime", "")
                reply_count = post.get("replyCount", 0)
                is_top = post.get("isTop", False)
                
                if is_top:
                    continue  # 跳过置顶
                
                print(f"{i}. [{create_time}] {title} —— @{author} ({reply_count} 回复)")
        else:
            print("❌ API 返回 re=false 或 result 为空")
            print("Response:", data)
    else:
        print("❌ HTTP 请求失败:", response.text)

except Exception as e:
    print("⚠️ 异常:", e)















































