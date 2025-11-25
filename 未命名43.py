import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 等待页面完全加载的函数
def wait_for_page_to_load():
    try:
        # 使用 WebDriverWait 等待页面加载完成
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except Exception as e:
        print(f"页面加载失败: {e}")

# 获取评论的函数
def load_all_comments(news_url):
    driver.get(news_url)
    time.sleep(2)  # 等待页面加载
    
    # 假设评论区域有“加载更多”按钮，模拟点击按钮
    while True:
        try:
            # 找到“加载更多”按钮并点击
            load_more_button = driver.find_element(By.CSS_SELECTOR, '.load-more-comments')  # 假设是这个类名
            load_more_button.click()
            time.sleep(2)  # 等待更多评论加载
        except Exception as e:
            print(f"没有更多评论按钮或加载完毕: {e}")
            break

    # 返回加载完的页面内容
    return driver.page_source

# 获取评论的详细信息
def get_comments(news_url):
    page_source = load_all_comments(news_url)
    
    soup = BeautifulSoup(page_source, 'lxml')
    comments = []
    
    # 假设评论在 class='comment-item' 中
    comment_elements = soup.find_all('div', class_='comment-item')
    for comment in comment_elements:
        username = comment.find('span', class_='comment-author').get_text(strip=True)
        content = comment.find('p', class_='comment-content').get_text(strip=True)
        comments.append({
            "username": username,
            "content": content
        })
    
    comment_count = len(comments)
    return comments, comment_count

# 抓取50页的新闻
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载
    wait_for_page_to_load()
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 默认评论数为0
            comment_count = 0
            comments = []
            comment_element = item.find('div', class_='level1_item')
            if comment_element:
                comment_count = int(comment_element.get('data-reply_count', 0))
                if comment_count > 0:
                    comments, comment_count = get_comments(link)
            
            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  
                "comments_count": comment_count,  
                "link": link,
                "comments": comments
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()






import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 抓取50页的新闻
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 默认评论数为0
            comment_count = 0
            comments = []
            comment_element = item.find('div', class_='level1_item')
            if comment_element:
                comment_count = int(comment_element.get('data-reply_count', 0))
                
                # 如果有评论，加载评论
                if comment_count > 0:
                    driver.get(link)
                    time.sleep(2)  # 等待页面加载
                    
                    # 模拟点击“加载更多”按钮，直到评论全部加载
                    while True:
                        try:
                            load_more_button = driver.find_element(By.CSS_SELECTOR, '.load-more-comments')  # 假设是这个类名
                            load_more_button.click()
                            time.sleep(2)  # 等待更多评论加载
                        except Exception as e:
                            print(f"没有更多评论按钮或加载完毕: {e}")
                            break
                    
                    # 获取评论区的HTML
                    soup = BeautifulSoup(driver.page_source, 'lxml')
                    comment_elements = soup.find_all('div', class_='comment-item')
                    for comment in comment_elements:
                        username = comment.find('span', class_='comment-author').get_text(strip=True)
                        content = comment.find('p', class_='comment-content').get_text(strip=True)
                        comments.append({
                            "username": username,
                            "content": content
                        })

            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  
                "comments_count": comment_count,  
                "link": link,
                "comments": comments
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments1.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments1.json")

# 关闭浏览器
driver.quit()









import time
import json
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 设置 Selenium WebDriver（确保 chromedriver 可用）
driver = webdriver.Chrome()

# 用于保存所有新闻的列表
all_news = []

# 抓取50页的新闻
for page in range(1, 51):  # 循环抓取50页
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    # 等待页面加载
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    soup = BeautifulSoup(driver.page_source, 'lxml')
    
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))
    
    if news_items:
        print(f"第{page}页抓取成功，共抓取到 {len(news_items)} 条新闻。")
        
        # 遍历每条新闻并提取数据
        for item in news_items:
            title = item.find('p', class_='title').get_text().strip()
            content = item.find('p', class_='info').get_text().strip()
            publish_time = item.find('p', class_='time').get_text().strip()
            link = item.find('a')['href']
            
            # 默认评论数为0
            comment_count = 0
            comments = []
            comment_element = item.find('div', class_='level1_item')
            if comment_element:
                comment_count = int(comment_element.get('data-reply_count', 0))
                
                # 如果有评论，加载评论
                if comment_count > 0:
                    driver.get(link)
                    time.sleep(2)  # 等待页面加载
                    
                    # 模拟点击“加载更多”按钮，直到评论全部加载
                    while True:
                        try:
                            load_more_button = driver.find_element(By.CSS_SELECTOR, '.load-more-comments')  # 假设是这个类名
                            load_more_button.click()
                            time.sleep(2)  # 等待更多评论加载
                        except Exception as e:
                            print(f"没有更多评论按钮或加载完毕: {e}")
                            break
                    
                    # 获取评论区的HTML
                    soup = BeautifulSoup(driver.page_source, 'lxml')
                    comment_elements = soup.find_all('div', class_='level2_item')
                    for comment in comment_elements:
                        # 获取评论者名称
                        username = comment.find('a', class_='replyer_name').get_text(strip=True)
                        # 获取评论内容
                        content = comment.find('span', class_='l2_short_text').get_text(strip=True)
                        # 获取评论时间和位置（如果需要）
                        publish_time = comment.find('span', class_='time').get_text(strip=True)
                        location = comment.find('span', class_='reply_ip').get_text(strip=True)
                        
                        comments.append({
                            "username": username,
                            "content": content,
                            "publish_time": publish_time,
                            "location": location
                        })

            # 构建新闻数据字典
            news_data = {
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "source": "EastMoney",  
                "comments_count": comment_count,  
                "link": link,
                "comments": comments
            }
            
            # 将新闻数据添加到总列表中
            all_news.append(news_data)
    
    else:
        print(f"第{page}页没有找到新闻条目，可能页面结构发生了变化。")
    
    # 每页抓取完成后等待 1 秒
    time.sleep(1)

# 完成所有页面抓取后，将数据保存为 JSON 文件
with open('news_data_with_comments.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

# 打印完成消息
print("所有数据已保存到 news_data_with_comments.json")

# 关闭浏览器
driver.quit()








import time
import json
import re
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# 提取新闻 ID（链接最后 10~12 位数字）
def extract_news_id(url):
    match = re.search(r'/(\d+)\.html', url)
    return match.group(1) if match else None



# 获取一级评论（含评论内容、点赞、发布时间等）
def get_level1_comments(news_id):
    comments = []
    page = 1

    while True:
        api = f"https://np-community.eastmoney.com/api/Comments/GetComments?postid={news_id}&p={page}&ps=30"
        response = requests.get(api).json()

        items = response.get("re", {}).get("comments", [])

        if not items:
            break

        for c in items:
            comments.append({
                "reply_id": c.get("replyid"),
                "username": c.get("uname"),
                "content": c.get("content"),
                "time": c.get("posttime"),
                "like_count": c.get("like"),
                "location": c.get("ipfrom"),
                "level2_count": c.get("count")  # 二级评论数量
            })

        page += 1
        time.sleep(0.3)

    return comments



# 获取二级评论（根据一级评论 reply_id）
def get_level2_comments(reply_id):
    api = f"https://np-community.eastmoney.com/api/Comments/GetLevel2Comments?replyid={reply_id}&ps=50"
    res = requests.get(api).json()

    items = res.get("re", {}).get("comments", [])
    result = []

    for c in items:
        result.append({
            "username": c.get("uname"),
            "content": c.get("content"),
            "time": c.get("posttime"),
            "location": c.get("ipfrom"),
            "like_count": c.get("like"),
        })

    return result



# Selenium 抓新闻列表
driver = webdriver.Chrome()
all_news = []


for page in range(1, 51):
    url = f"https://finance.eastmoney.com/a/ccjdd.html?page={page}"
    driver.get(url)
    
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    soup = BeautifulSoup(driver.page_source, 'lxml')
    news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))

    print(f"第 {page} 页，共 {len(news_items)} 条")

    for item in news_items:
        title = item.find('p', class_='title').get_text(strip=True)
        summary = item.find('p', class_='info').get_text(strip=True)
        publish_time = item.find('p', class_='time').get_text(strip=True)
        link = item.find('a')['href']

        # 提取新闻 ID
        news_id = extract_news_id(link)

        # 如果无法提取 ID 直接跳过
        if not news_id:
            print(f"跳过：无法提取 ID → {link}")
            continue

        print(f"抓取评论 newsID={news_id}")

        # 一级评论
        level1 = get_level1_comments(news_id)

        # 二级评论
        for c in level1:
            c["level2"] = get_level2_comments(c["reply_id"])

        all_news.append({
            "title": title,
            "summary": summary,
            "publish_time": publish_time,
            "link": link,
            "news_id": news_id,
            "comments": level1
        })

    time.sleep(1)


driver.quit()

# 保存 JSON
with open("news_with_comments.json", "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

print("数据已保存：news_with_comments.json")














"""
eastmoney_finance_comments.py

功能：
- 抓取 finance.eastmoney.com 指定页数的新闻列表（默认 50 页）
- 从新闻链接提取 news_id
- 通过评论 API 批量抓取一级评论（分页）和对应的二级评论（回复）
- 支持并发（可配置）、重试、退避、日志、结果保存为 JSON

使用：
python eastmoney_finance_comments.py
"""

import re
import time
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# -------- 配置区 --------
BASE_LIST_URL = "https://finance.eastmoney.com/a/ccjdd.html?page={page}"  # 财经频道示例（你的页面 URL 模式）
PAGES_TO_SCRAPE = 50  # 你确认的：抓取 50 页
OUTPUT_FILE = "eastmoney_finance_50pages_with_comments.json"

# 并发与速率
MAX_WORKERS = 4  # 并发抓新闻对应评论的线程数（谨慎调大以免被封）
REQUESTS_SLEEP = 0.3  # 每个评论 API 请求间的最小间隔（秒）
PAGE_SLEEP = 1.0  # 抓取新闻列表每页间隔（秒）

# requests 重试参数
RETRY_TIMES = 3
RETRY_BACKOFF = 1.0  # 基本退避（乘以 attempt）

# 可选代理（若被封可以启用）示例格式：{"http": "http://user:pass@host:port", "https": "..."}
PROXIES = None  # 或配置为字典

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eastmoney_scraper")

# 一组可用 User-Agent，程序会随机选一个以降低被识别概率
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

# -------- 工具函数 --------
def make_headers(referer: Optional[str] = None) -> Dict[str, str]:
    h = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Origin": "https://finance.eastmoney.com",
        "Host": "np-comment.eastmoney.com",
    }
    if referer:
        h["Referer"] = referer
    return h

def extract_news_id(url: str) -> Optional[str]:
    """
    从新闻链接中提取新闻 ID（匹配 /digits+.html）。
    返回数字字符串或 None。
    """
    m = re.search(r'/(\d+)\.html', url)
    return m.group(1) if m else None

def safe_json(res: requests.Response) -> Optional[Dict[str, Any]]:
    """
    尝试解析 JSON；若不是 JSON 或出现反爬，返回 None 并打印前一段文本。
    """
    try:
        return res.json()
    except Exception:
        snippet = res.text[:200].replace("\n", " ")
        logger.warning("Response is not valid JSON (可能被反爬). 前200字符: %s", snippet)
        return None

# -------- API 抓取函数（带重试） --------
def _get_with_retry(session: requests.Session, url: str, headers: Dict[str,str], params=None, proxies=None) -> Optional[requests.Response]:
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=15, proxies=proxies)
            # 简单的状态码判断
            if r.status_code == 200:
                return r
            else:
                logger.warning("请求 %s 时返回状态 %s", url, r.status_code)
        except requests.RequestException as e:
            logger.warning("请求异常（attempt %d/%d）: %s", attempt, RETRY_TIMES, e)
        time.sleep(RETRY_BACKOFF * attempt + random.random() * 0.5)
    logger.error("多次重试失败：%s", url)
    return None

def get_level1_comments(session: requests.Session, news_id: str) -> List[Dict[str, Any]]:
    """
    抓取一级评论（分页直到空结果）。
    使用接口：
    https://np-comment.eastmoney.com/api/comment/GetNewsComments?newsId={news_id}&pageIndex={page}&pageSize={page_size}
    返回 JSON 的评论数组字段可能不同，本函数会尽量兼容常见字段名。
    """
    comments = []
    page = 0
    page_size = 20

    while True:
        api = "https://np-comment.eastmoney.com/api/comment/GetNewsComments"
        headers = make_headers(referer=f"https://finance.eastmoney.com/a/{news_id}.html")
        params = {"newsId": news_id, "pageIndex": page, "pageSize": page_size}
        r = _get_with_retry(session, api, headers=headers, params=params, proxies=PROXIES)
        if r is None:
            # 该页请求失败，终止或跳过
            logger.error("无法获取一级评论（news_id=%s page=%d）", news_id, page)
            break

        data = safe_json(r)
        if data is None:
            # 非 JSON（可能被反爬）
            break

        # 兼容多种结构：一些接口把数组放在 data['comments']、data['re']['comments']、data['data'] 等
        items = None
        for key in ("comments", "data", "re", "result"):
            if isinstance(data.get(key), list):
                items = data.get(key)
                break
            # 有时 data['re'] 是 dict 包含 'comments'
            if isinstance(data.get(key), dict):
                for sub in ("comments", "data", "list"):
                    if isinstance(data[key].get(sub), list):
                        items = data[key].get(sub)
                        break
                if items is not None:
                    break

        # 直接检查 top-level list (some APIs return list directly)
        if items is None and isinstance(data, list):
            items = data

        if not items:
            # 没有更多评论了
            break

        # 解析每个 item，尽量提取常见字段
        for it in items:
            # 字段推断器
            comment_id = None
            for fld in ("commentId", "commentid", "replyid", "replyId", "id", "ID"):
                if fld in it:
                    comment_id = it.get(fld)
                    break

            username = it.get("uname") or it.get("userName") or it.get("nickName") or it.get("nickname") or it.get("user_name") or it.get("username") or ""
            content = it.get("content") or it.get("body") or it.get("comment") or ""
            posttime = it.get("posttime") or it.get("time") or it.get("createTime") or it.get("publish_time") or ""
            like = it.get("like") or it.get("likes") or it.get("agree") or it.get("up") or 0
            ipfrom = it.get("ipfrom") or it.get("location") or ""

            level2_count = it.get("count") or it.get("replyCount") or it.get("childCount") or 0

            comment_obj = {
                "comment_id": comment_id,
                "username": username,
                "content": content,
                "time": posttime,
                "like_count": like,
                "location": ipfrom,
                "level2_count": level2_count,
                "raw": it
            }
            comments.append(comment_obj)

        page += 1
        time.sleep(REQUESTS_SLEEP + random.random() * 0.2)

    return comments

def get_level2_comments(session: requests.Session, news_id: str, comment_id: Any) -> List[Dict[str, Any]]:
    """
    抓取二级评论（回复）：
    https://np-comment.eastmoney.com/api/comment/GetReplyComments?newsId={news_id}&commentId={comment_id}&pageIndex={page}&pageSize={page_size}
    """
    if not comment_id:
        return []
    replies = []
    page = 0
    page_size = 20

    while True:
        api = "https://np-comment.eastmoney.com/api/comment/GetReplyComments"
        headers = make_headers(referer=f"https://finance.eastmoney.com/a/{news_id}.html")
        params = {"newsId": news_id, "commentId": comment_id, "pageIndex": page, "pageSize": page_size}
        r = _get_with_retry(session, api, headers=headers, params=params, proxies=PROXIES)
        if r is None:
            logger.error("无法获取二级评论（news_id=%s comment_id=%s page=%d）", news_id, comment_id, page)
            break

        data = safe_json(r)
        if data is None:
            break

        # 解析同样兼容多种字段名
        items = None
        for key in ("comments", "data", "re", "result"):
            if isinstance(data.get(key), list):
                items = data.get(key)
                break
            if isinstance(data.get(key), dict):
                for sub in ("comments", "data", "list"):
                    if isinstance(data[key].get(sub), list):
                        items = data[key].get(sub)
                        break
                if items is not None:
                    break

        if items is None and isinstance(data, list):
            items = data

        if not items:
            break

        for it in items:
            username = it.get("uname") or it.get("userName") or it.get("nickName") or ""
            content = it.get("content") or ""
            posttime = it.get("posttime") or it.get("time") or ""
            like = it.get("like") or 0
            ipfrom = it.get("ipfrom") or ""

            replies.append({
                "username": username,
                "content": content,
                "time": posttime,
                "like_count": like,
                "location": ipfrom,
                "raw": it
            })

        page += 1
        time.sleep(REQUESTS_SLEEP + random.random() * 0.2)

    return replies

# -------- 列表页抓取（requests + bs4） --------
def fetch_news_list_page(session: requests.Session, page: int) -> List[Dict[str, Any]]:
    url = BASE_LIST_URL.format(page=page)
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://finance.eastmoney.com/",
    }
    r = _get_with_retry(session, url, headers=headers, proxies=PROXIES)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    # 页面中新闻项在 li 元素，id 以 newsTr 开头（你之前脚本使用的选择器）
    items = soup.find_all("li", id=lambda x: x and x.startswith("newsTr"))
    results = []

    for it in items:
        # 标题、摘要、时间、链接的定位可能有变化，这里做尽量通用的提取
        title_tag = it.find("p", class_="title")
        info_tag = it.find("p", class_="info")
        time_tag = it.find("p", class_="time")
        a_tag = it.find("a", href=True)

        title = title_tag.get_text(strip=True) if title_tag else (a_tag.get_text(strip=True) if a_tag else "")
        summary = info_tag.get_text(strip=True) if info_tag else ""
        publish_time = time_tag.get_text(strip=True) if time_tag else ""
        link = a_tag["href"].strip() if a_tag else ""

        # 相对链接变成绝对（若必要）
        if link.startswith("//"):
            link = "https:" + link
        elif link.startswith("/"):
            link = "https://finance.eastmoney.com" + link

        news_id = extract_news_id(link)

        results.append({
            "title": title,
            "summary": summary,
            "publish_time": publish_time,
            "link": link,
            "news_id": news_id
        })

    return results

# -------- 主流程 --------
def process_news_item(session: requests.Session, news_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单条新闻：根据 news_id 抓评论（一级 + 二级）
    """
    news_id = news_item.get("news_id")
    if not news_id:
        logger.warning("跳过：无法提取 ID → %s", news_item.get("link"))
        news_item["comments"] = []
        return news_item

    # 抓一级评论
    logger.info("抓取评论 newsID=%s 标题=%s", news_id, news_item.get("title")[:30])
    level1 = get_level1_comments(session, news_id)

    # 抓二级评论（并发内部一级评论的二级拉取）
    if level1:
        for c in level1:
            cid = c.get("comment_id")
            # 兼容：如果 comment_id 是 None，但 raw 中存在 id 字段，尝试获取
            if not cid and isinstance(c.get("raw"), dict):
                for alt in ("commentId","commentid","replyid","replyId","id","ID"):
                    if alt in c["raw"]:
                        cid = c["raw"].get(alt)
                        c["comment_id"] = cid
                        break
            c["level2"] = get_level2_comments(session, news_id, cid)
    else:
        logger.info("newsID=%s 无一级评论或获取失败。", news_id)

    news_item["comments"] = level1
    return news_item

def main():
    session = requests.Session()
    all_news: List[Dict[str, Any]] = []

    # 先抓取新闻列表（按页），收集 news items
    logger.info("开始抓取新闻列表（%d 页）", PAGES_TO_SCRAPE)
    for page in range(1, PAGES_TO_SCRAPE + 1):
        try:
            page_items = fetch_news_list_page(session, page)
            logger.info("第 %d 页抓取到 %d 条", page, len(page_items))
            all_news.extend(page_items)
        except Exception as e:
            logger.exception("抓取第 %d 页异常：%s", page, e)
        time.sleep(PAGE_SLEEP + random.random() * 0.5)

    logger.info("收集到总新闻条数：%d", len(all_news))

    # 使用线程池并发抓取每条新闻的评论（谨慎的并发数）
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_news_item, session, ni) for ni in all_news]
        for f in tqdm(as_completed(futures), total=len(futures), desc="抓取评论进度"):
            try:
                res = f.result()
                results.append(res)
            except Exception as e:
                logger.exception("单个新闻处理失败：%s", e)

    # 保存为 JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)

    logger.info("完成，保存到 %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()












import requests
from bs4 import BeautifulSoup
import random
import json
import time
import logging
import re

# ===============================
# 日志设置
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ===============================
# 浏览器 UA 池
# ===============================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:108.0) Gecko/20100101 Firefox/108.0",
]

# ===============================
# 1. 构造财经频道分页 URL
# ===============================
def build_list_url(page: int) -> str:
    if page == 1:
        return "https://finance.eastmoney.com/a/cgnjj.html"   # 财经要闻首页
    return f"https://finance.eastmoney.com/a/cgnjj_{page}.html"


# ===============================
# 2. 提取新闻 ID（非常关键！）
# ===============================
def extract_news_id(url: str):
    match = re.search(r'/(\d+)\.html', url)
    return match.group(1) if match else None


# ===============================
# 3. 获取评论（一级评论）
# ===============================
def get_level1_comments(news_id: str):
    url = (
        f"https://np-comment.eastmoney.com/api/comment/GetNewsComments"
        f"?newsId={news_id}&pageIndex=0&pageSize=50"
    )

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": f"https://finance.eastmoney.com/a/{news_id}.html",
        "Accept": "application/json, text/plain, */*",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return data.get("comments", [])

    except Exception as e:
        logging.warning(f"一级评论获取失败 newsId={news_id} → {e}")
        return []


# ===============================
# 4. 获取二级评论
# ===============================
def get_level2_comments(news_id: str, comment_id: str):
    url = (
        f"https://np-comment.eastmoney.com/api/comment/GetReplyComments"
        f"?newsId={news_id}&commentId={comment_id}&pageIndex=0&pageSize=50"
    )

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": f"https://finance.eastmoney.com/a/{news_id}.html",
        "Accept": "application/json, text/plain, */*",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return data.get("comments", [])

    except Exception as e:
        logging.warning(f"二级评论获取失败 commentId={comment_id} → {e}")
        return []


# ===============================
# 5. 抓取新闻列表（每页）
# ===============================
def fetch_news_list(page: int):
    url = build_list_url(page)

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://finance.eastmoney.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")

        items = soup.find_all("li", id=lambda x: x and x.startswith("newsTr"))
        logging.info(f"第 {page} 页抓取到 {len(items)} 条")

        news_list = []
        for li in items:
            title = li.find("p", class_="title").get_text(strip=True)
            summary = li.find("p", class_="info").get_text(strip=True)
            time_str = li.find("p", class_="time").get_text(strip=True)
            link = li.find("a").get("href")

            news_id = extract_news_id(link)
            if not news_id:
                logging.warning(f"跳过：无法获取新闻ID → {link}")
                continue

            news_list.append({
                "news_id": news_id,
                "title": title,
                "summary": summary,
                "time": time_str,
                "link": link
            })

        return news_list

    except Exception as e:
        logging.error(f"第 {page} 页抓取失败：{e}")
        return []


# ===============================
# 主程序入口
# ===============================
def main():
    all_data = []
    TOTAL_PAGES = 50

    logging.info(f"开始抓取财经频道，共 {TOTAL_PAGES} 页")

    for page in range(1, TOTAL_PAGES + 1):
        news_page = fetch_news_list(page)

        for news in news_page:
            news_id = news["news_id"]
            logging.info(f"抓取评论 newsID={news_id}")

            # 一级评论
            level1 = get_level1_comments(news_id)

            # 添加二级评论
            for c1 in level1:
                cid = str(c1.get("comment_id"))
                c1["replies"] = get_level2_comments(news_id, cid)

            news["comments"] = level1
            all_data.append(news)

        time.sleep(1)

    # 保存 JSON
    with open("finance_news_with_comments.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    logging.info("抓取完成 → finance_news_with_comments.json")


if __name__ == "__main__":
    main()






import requests
import random
import json
import time
import logging
import re

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# User-Agent池
UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/119.0",
]

# ===============================
# 获取财经新闻列表（真正的数据源）
# ===============================
def fetch_news_list(page):

    url = f"https://finance.eastmoney.com/api/article/list?channel=caijing&sort=time&page={page}&limit=20"

    headers = {"User-Agent": random.choice(UA)}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        items = data.get("data", {}).get("articles", [])
        logging.info(f"第 {page} 页抓取到 {len(items)} 条新闻")

        news_list = []

        for item in items:
            news_id = str(item.get("newsId"))
            link = item.get("url")
            title = item.get("title")
            summary = item.get("summary")
            time_str = item.get("showTime")

            news_list.append({
                "news_id": news_id,
                "title": title,
                "summary": summary,
                "time": time_str,
                "link": link
            })

        return news_list

    except Exception as e:
        logging.error(f"抓取新闻列表失败 page={page} → {e}")
        return []


# ===============================
# 一级评论
# ===============================
def get_level1_comments(news_id):

    api = (
        "https://np-comment.eastmoney.com/api/comment/GetNewsComments"
        f"?newsId={news_id}&pageIndex=0&pageSize=50"
    )

    headers = {
        "User-Agent": random.choice(UA),
        "Referer": f"https://finance.eastmoney.com/a/{news_id}.html"
    }

    try:
        r = requests.get(api, headers=headers, timeout=10)
        data = r.json()
        return data.get("comments", [])

    except Exception as e:
        logging.warning(f"一级评论获取失败 newsId={news_id} → {e}")
        return []


# ===============================
# 二级评论
# ===============================
def get_level2_comments(news_id, comment_id):

    api = (
        "https://np-comment.eastmoney.com/api/comment/GetReplyComments"
        f"?newsId={news_id}&commentId={comment_id}&pageIndex=0&pageSize=50"
    )

    headers = {"User-Agent": random.choice(UA)}

    try:
        r = requests.get(api, headers=headers, timeout=10)
        data = r.json()
        return data.get("comments", [])

    except Exception as e:
        logging.warning(f"二级评论失败 commentId={comment_id} → {e}")
        return []


# ===============================
# 主程序
# ===============================
def main():
    TOTAL_PAGES = 50
    all_news = []

    logging.info(f"开始抓取财经频道，共 {TOTAL_PAGES} 页")

    for page in range(1, TOTAL_PAGES + 1):

        news_list = fetch_news_list(page)

        for n in news_list:
            news_id = n["news_id"]
            logging.info(f"抓取评论 newsID={news_id}")

            level1 = get_level1_comments(news_id)

            # 抓取二级评论
            for c in level1:
                cid = str(c.get("comment_id"))
                c["replies"] = get_level2_comments(news_id, cid)

            n["comments"] = level1
            all_news.append(n)

        time.sleep(1)

    # 保存
    with open("finance_news.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=4)

    logging.info("抓取完成 → finance_news.json")


if __name__ == "__main__":
    main()
















import requests
import random
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


# =========================
# 获取财经新闻列表（最终真实接口）
# =========================
def fetch_news_list(page):
    url = f"https://emdata.eastmoney.com/news/info/list?page={page}&limit=20&type=1"

    headers = {
        "User-Agent": random.choice(UA),
        "Referer": "https://finance.eastmoney.com",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        items = data.get("data", [])
        logging.info(f"第 {page} 页抓取到 {len(items)} 条新闻")

        return items

    except Exception as e:
        logging.error(f"抓取新闻列表失败 page={page} → {e}")
        logging.error(f"返回内容: {r.text[:200]}")
        return []


# =========================
# 获取一级评论
# =========================
def get_level1_comments(news_id):
    api = (
        "https://np-comment.eastmoney.com/api/comment/GetNewsComments"
        f"?newsId={news_id}&pageIndex=0&pageSize=50"
    )

    headers = {
        "User-Agent": random.choice(UA),
        "Referer": f"https://finance.eastmoney.com/a/{news_id}.html",
    }

    try:
        r = requests.get(api, headers=headers, timeout=10)
        return r.json().get("comments", [])

    except Exception as e:
        logging.warning(f"一级评论失败 newsId={news_id} → {e}")
        return []


# =========================
# 获取二级评论
# =========================
def get_level2_comments(news_id, comment_id):
    api = (
        "https://np-comment.eastmoney.com/api/comment/GetReplyComments"
        f"?newsId={news_id}&commentId={comment_id}&pageIndex=0&pageSize=50"
    )

    headers = {"User-Agent": random.choice(UA)}

    try:
        r = requests.get(api, headers=headers, timeout=10)
        return r.json().get("comments", [])

    except Exception as e:
        logging.warning(f"二级评论失败 commentId={comment_id} → {e}")
        return []


# =========================
# 主程序
# =========================
def main():
    TOTAL_PAGES = 50
    all_news = []

    logging.info(f"开始抓取财经频道，共 {TOTAL_PAGES} 页")

    for page in range(1, TOTAL_PAGES + 1):

        news_list = fetch_news_list(page)

        for n in news_list:
            news_id = str(n.get("id"))
            logging.info(f"抓取评论 newsID={news_id}")

            c1 = get_level1_comments(news_id)

            # 获取二级评论
            for c in c1:
                cid = str(c.get("comment_id"))
                c["replies"] = get_level2_comments(news_id, cid)

            n["comments"] = c1
            all_news.append(n)

        time.sleep(1)

    # 保存
    with open("finance_news.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=4)

    logging.info("抓取完成 → finance_news.json")


if __name__ == "__main__":
    main()












import requests
import logging
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


# =====================================================
#   步骤 1：使用 Session 自动获取东财 Cookie（必须）
# =====================================================
def get_session():
    s = requests.Session()

    headers = {
        "User-Agent": random.choice(UA),
        "Referer": "https://finance.eastmoney.com",
    }

    # 第一次访问主页 → 自动下发 cookie（必需）
    s.get("https://finance.eastmoney.com", headers=headers, timeout=10)

    logging.info(f"获取到 Cookie → {s.cookies.get_dict()}")
    return s


# =====================================================
#   步骤 2：使用拿到的 cookie 请求新闻列表（就不会被返回 HTML）
# =====================================================
def fetch_news_list(session, page):
    url = f"https://emdata.eastmoney.com/news/info/list?page={page}&limit=20&type=1"

    headers = {
        "User-Agent": random.choice(UA),
        "Referer": "https://finance.eastmoney.com",
    }

    r = session.get(url, headers=headers, timeout=10)

    # 如果返回 HTML，就说明 cookie 失效
    if r.text.startswith("<!DOCTYPE html"):
        logging.error(f"❌ 被反爬，返回 HTML page={page}")
        logging.error(f"前 200 字节：{r.text[:200]}")
        return None

    try:
        data = r.json()
        return data.get("data", [])

    except Exception as e:
        logging.error(f"JSON 解析失败 page={page} → {e}")
        logging.error(f"返回内容：{r.text[:200]}")
        return None


# =====================================================
#   主程序
# =====================================================
def main():
    session = get_session()

    logging.info("开始抓取财经频道，共 50 页")

    for page in range(1, 51):

        items = fetch_news_list(session, page)

        if items is None:
            logging.error(f"第 {page} 页失败（反爬），已跳过")
            continue

        logging.info(f"第 {page} 页抓取到 {len(items)} 条")

        time.sleep(1)


if __name__ == "__main__":
    main()






import requests
import json
import time
import logging

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 你的 Cookie（直接复制你提供的）
COOKIE = "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; nid_create_time=1762410811391; gvi=feRT-qrGn2zkzSvVr-lYCef07; gvi_create_time=1762410811391; qgqp_b_id=784e4fb0f790a7016edd65e07c75bb2c; fullscreengg=1; fullscreengg2=1; st_si=53814237522125; st_asi=delete; st_pvi=65571146956567; st_sp=2025-11-06%2014%3A33%3A31; st_inirUrl=https%3A%2F%2Fchatgpt.com%2F; st_sn=5; st_psi=20251119173600913-113104312931-0001230916"

HEADERS = {
    "Cookie": COOKIE,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://finance.eastmoney.com/a/ccjdd.html"
}

# 财经频道新闻列表接口
NEWS_LIST_URL = "https://finance.eastmoney.com/a/ccjdd.html?page={page}"

# 保存新闻
all_news = []

for page in range(1, 51):
    logging.info(f"抓取财经频道，第 {page} 页")
    try:
        response = requests.get(NEWS_LIST_URL.format(page=page), headers=HEADERS, timeout=10)
        if response.status_code != 200:
            logging.error(f"第 {page} 页请求失败，status_code={response.status_code}")
            continue

        # 注意：页面是 HTML，需要解析
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'lxml')
        news_items = soup.find_all('li', id=lambda x: x and x.startswith('newsTr'))

        logging.info(f"第 {page} 页抓取到 {len(news_items)} 条新闻")

        for item in news_items:
            try:
                title = item.find('p', class_='title').get_text(strip=True)
                link = item.find('a')['href']
                content = item.find('p', class_='info').get_text(strip=True)
                publish_time = item.find('p', class_='time').get_text(strip=True)

                # 获取新闻 ID
                news_id = link.split('/')[-1].replace('.html','')

                # 获取评论接口
                COMMENT_API = f"https://guba.eastmoney.com/comment/list?type=1&newsid={news_id}&page=1&_=1"
                comments = []
                comment_resp = requests.get(COMMENT_API, headers=HEADERS, timeout=10)
                if comment_resp.status_code == 200:
                    try:
                        comment_json = comment_resp.json()
                        # 遍历评论
                        for com in comment_json.get('data', {}).get('comments', []):
                            comments.append({
                                "username": com.get('nickname'),
                                "content": com.get('content'),
                                "like_count": com.get('likeCount', 0)
                            })
                    except Exception as e:
                        logging.warning(f"解析评论失败 newsID={news_id} → {e}")

                news_data = {
                    "title": title,
                    "link": link,
                    "content": content,
                    "publish_time": publish_time,
                    "news_id": news_id,
                    "comments": comments
                }
                all_news.append(news_data)

                time.sleep(0.5)  # 避免请求过快

            except Exception as e:
                logging.warning(f"解析新闻失败 → {e}")

        time.sleep(1)  # 每页间隔

    except Exception as e:
        logging.error(f"抓取新闻列表失败 page={page} → {e}")

# 保存为 JSON
with open('eastmoney_finance_news.json', 'w', encoding='utf-8') as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

logging.info("所有数据抓取完成！")

















import requests
import json
import time
import re

news_id = "202511173566214160"
timestamp = int(time.time() * 1000)
url = f"https://gbapi.eastmoney.com/reply/JSONP/ArticleHotReply?callback=jQuery123456789&plat=web&version=300&product=guba&h=a73bfb37b81c3b05b8f7b22bbfc11422&postid={news_id}&type=1&_={timestamp}"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Referer": f"https://finance.eastmoney.com/a/ccjdd.html",
    "Cookie": "st_nvi=xxx; nid=xxx; ...",  # 填写你抓到的 Cookie
}

res = requests.get(url, headers=headers)
text = res.text

# 去掉 JSONP 包裹
json_str = re.search(r'\((.*)\)', text).group(1)
data = json.loads(json_str)

# 一级评论列表
comments = data.get("Data", {}).get("Replies", [])
for c in comments:
    print(c["NickName"], c["Content"])







import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging

# ----------------- 日志设置 -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------- 配置 -----------------
BASE_URL = "https://finance.eastmoney.com/a/ccjdd.html?page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Referer": "https://finance.eastmoney.com/a/ccjdd.html",
    "Cookie": "st_nvi=m42mefF8u9WXMqIiGgxsk321d; nid=0e17cb22ecf6960f4858bfd8cbdced17; gvi=feRT-qrGn2zkzSvVr-lYCef07; qgqp_b_id=784e4fb0f790a7016edd65e07"  # 填入你获取的 Cookie
}

# JSONP 评论接口模板
COMMENT_URL = "https://gbapi.eastmoney.com/reply/JSONP/ArticleHotReply?callback=jQuery123456789&plat=web&version=300&product=guba&h=a73bfb37b81c3b05b8f7b22bbfc11422&postid={news_id}&type=1&_={ts}"

# ----------------- 函数 -----------------
def get_news_list(page):
    """抓取财经频道单页新闻列表"""
    url = BASE_URL.format(page)
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        news_items = soup.find_all("li", id=lambda x: x and x.startswith("newsTr"))
        news_list = []
        for item in news_items:
            try:
                title_tag = item.find("p", class_="title")
                info_tag = item.find("p", class_="info")
                time_tag = item.find("p", class_="time")
                link_tag = item.find("a")

                news_id = link_tag["href"].split("/")[-1].replace(".html", "")
                news_list.append({
                    "title": title_tag.get_text(strip=True) if title_tag else "",
                    "content": info_tag.get_text(strip=True) if info_tag else "",
                    "publish_time": time_tag.get_text(strip=True) if time_tag else "",
                    "link": link_tag["href"],
                    "news_id": news_id
                })
            except Exception as e:
                logging.warning(f"跳过：无法解析新闻条目 → {link_tag['href'] if link_tag else '未知链接'} ({e})")
        return news_list
    except Exception as e:
        logging.error(f"抓取新闻列表失败 page={page} → {e}")
        return []

def get_comments(news_id):
    """抓取单条新闻的一级评论"""
    ts = int(time.time() * 1000)
    url = COMMENT_URL.format(news_id=news_id, ts=ts)
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        text = res.text
        # 去掉 JSONP 包裹
        json_str = re.search(r'\((.*)\)', text).group(1)
        data = json.loads(json_str)
        replies = data.get("Data", {}).get("Replies", [])
        comments = [{"username": r["NickName"], "content": r["Content"]} for r in replies]
        return comments
    except Exception as e:
        logging.error(f"抓取评论 newsID={news_id} 失败 → {e}")
        return []

# ----------------- 主流程 -----------------
all_news = []

logging.info("开始抓取财经频道，共 50 页")
for page in range(1, 51):
    logging.info(f"抓取财经频道，第 {page} 页")
    news_list = get_news_list(page)
    logging.info(f"第 {page} 页抓取到 {len(news_list)} 条新闻")

    for news in news_list:
        news_id = news["news_id"]
        logging.info(f"抓取评论 newsID={news_id}")
        news["comments"] = get_comments(news_id)
        time.sleep(0.5)  # 防止请求过快

    all_news.extend(news_list)
    time.sleep(1)  # 每页等待 1 秒

# ----------------- 保存为 JSON -----------------
with open("eastmoney_finance_news.json", "w", encoding="utf-8") as f:
    json.dump(all_news, f, ensure_ascii=False, indent=4)

logging.info("全部数据抓取完成，已保存到 eastmoney_finance_news.json")

























