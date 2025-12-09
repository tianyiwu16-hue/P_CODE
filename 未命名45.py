import time
import json
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import re

# ---- 配置 ----
driver = webdriver.Chrome()  # Selenium WebDriver
all_news = []

# 可选：请求评论接口需要的 Headers（可以包含你的 Cookie）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Referer": "https://finance.eastmoney.com/a/ccjdd.html",
    "Cookie": "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; ..."  # 填入你的 Cookie
}

# ---- 函数：抓取评论 ----
def get_comments(news_id):
    """通过新闻ID抓取评论数据"""
    comments = []

    # JSONP接口URL模板
    url = f"https://gbapi.eastmoney.com/reply/JSONP/ArticleHotReply?callback=jQuery&_plat=web&version=300&product=guba&postid={news_id}&type=1&_={int(time.time()*1000)}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        text = resp.text

        # JSONP -> JSON
        json_text = re.sub(r'^jQuery\d+_\d+\((.*)\)$', r'\1', text)
        data = json.loads(json_text)

        for item in data.get('data', {}).get('list', []):
            comments.append({
                "username": item.get("nick", ""),
                "content": item.get("content", ""),
                "like_count": item.get("likecount", 0),
                "publish_time": item.get("time", "")
            })
    except Exception as e:
        print(f"抓取评论失败 newsID={news_id} → {e}")

    return comments

# ---- 主循环：抓取新闻列表 ----
for page in range(1, 51):
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    print(f"抓取财经频道，第 {page} 页")
    driver.get(url)
    driver.implicitly_wait(5)

    soup = BeautifulSoup(driver.page_source, "lxml")
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))

    if not news_items:
        print(f"第 {page} 页没有找到新闻条目")
        continue

    for item in news_items:
        title = item.find('p', class_='title').get_text(strip=True)
        content = item.find('p', class_='info').get_text(strip=True)
        publish_time = item.find('p', class_='time').get_text(strip=True)
        link = item.find('a')['href']

        # 获取新闻ID（URL里最后一段数字）
        match = re.search(r'/a/(\d+)\.html', link)
        if not match:
            print(f"跳过：无法提取 ID → {link}")
            continue
        news_id = match.group(1)

        # 抓评论
        comments = get_comments(news_id)
        comments_count = len(comments)

        # 构建新闻数据
        news_data = {
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "source": "EastMoney",
            "link": link,
            "comments_count": comments_count,
            "comments": comments
        }
        all_news.append(news_data)

    time.sleep(1)  # 防止被封

# ---- 保存 ----
with open("news_data_with_comments.json", "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

driver.quit()
print("所有数据已保存到 news_data_with_comments.json")















import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup

# -----------------------------
# 配置
# -----------------------------
driver = webdriver.Chrome()  # 请确保 chromedriver 可用
BASE_URL = "https://finance.eastmoney.com/a/ccjdd.html?page={}"
TOTAL_PAGES = 5  # 抓取页数
all_news = []

# -----------------------------
# 循环抓新闻
# -----------------------------
for page in range(1, TOTAL_PAGES + 1):
    print(f"抓取财经频道，第 {page} 页")
    driver.get(BASE_URL.format(page))
    driver.implicitly_wait(5)  # 等待页面加载
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if not news_items:
        print(f"第 {page} 页没有找到新闻条目")
        continue
    
    for item in news_items:
        title = item.find('p', class_='title').get_text(strip=True)
        content = item.find('p', class_='info').get_text(strip=True)
        publish_time = item.find('p', class_='time').get_text(strip=True)
        link = item.find('a')['href']
        
        # -----------------------------
        # 抓评论
        # -----------------------------
        comments = []
        driver.get(link)
        driver.implicitly_wait(5)
        soup_news = BeautifulSoup(driver.page_source, 'lxml')
        
        # 一级评论
        comment_divs = soup_news.select('div.level1_item')
        for c in comment_divs:
            user = c.select_one('a.replyer_name').get_text(strip=True)
            c_text = c.select_one('div.level1_reply_cont > div.short_text').get_text(strip=True)
            c_time = c.select_one('div.publish_time > span').get_text(strip=True)
            comments.append({
                "user": user,
                "content": c_text,
                "publish_time": c_time
            })
        
        news_data = {
            "title": title,
            "content": content,
            "publish_time": publish_time,
            "source": "EastMoney",
            "link": link,
            "comments": comments
        }
        all_news.append(news_data)
    
    # 避免被封，加延迟
    time.sleep(1)

# -----------------------------
# 保存 JSON
# -----------------------------
with open('finance_news_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

print("抓取完成，已保存到 finance_news_with_comments.json")
driver.quit()












import requests
import json

# 假设 postid=1627355623 对应股票 600000（请根据实际情况修改）
stock_code = "600000"
post_id = "1627355623"

headers = {
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': 'https://guba.eastmoney.com',
    'Referer': f'https://guba.eastmoney.com/news,{stock_code},{post_id}.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'sec-ch-ua': '"Chromium";v="127", "Not)A;Brand";v="24", "Google Chrome";v="127"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}

cookies_str = "your_real_cookie_here"  # ← 必须用真实有效的 Cookie！
cookies_dict = {k.strip(): v.strip() for k, v in (item.split('=', 1) for item in cookies_str.split(';') if '=' in item)}

payload = {
    'code': 'cjpl',
    'path': 'reply/api/Reply/ArticleNewReplyList',
    'param': json.dumps({
        'postid': post_id,
        'sort': 1,
        'sorttype': 1,
        'p': 1,
        'ps': 30
    }, separators=(',', ':')),
    'plat': 'Web',
    'env': '2',
    'origin': '',
    'version': '2022',
    'product': 'Guba'
}

response = requests.post(
    'https://guba.eastmoney.com/api/getData',
    data=payload,
    headers=headers,
    cookies=cookies_dict,
    timeout=10
)

print(response.status_code)
print(response.text[:500])





























