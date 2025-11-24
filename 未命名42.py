import requests
import json

url = "https://mguba.eastmoney.com/mguba2020/interface/GetData.aspx"

# 尝试大盘情绪指数
payload = {
    "app": "web",
    "client": "web",
    "version": "1.0",
    "action": "GubaEmotionIndex",
    "code": "0"  # 大盘
}

# 方法1：param 作为 GET 查询参数（明文 JSON 字符串）
params = {
    "param": json.dumps(payload, separators=(',', ':'))  # 去掉空格压缩
}

headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Referer": "https://mguba.eastmoney.com/",
}

response = requests.get(url, params=params, headers=headers)
print(response.text)





















