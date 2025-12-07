# ...existing code...
import requests
import json

headers = {
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            # 'Cookie': 'qgqp_b_id=2c741d207879f0ab1f0cb309ebe5a475; EMFUND1=null; EMFUND2=null; EMFUND3=null; EMFUND4=null; EMFUND5=null; EMFUND6=null; EMFUND7=null; EMFUND0=null; EMFUND9=07-23%2020%3A29%3A43@%23%24%u56FD%u6CF0%u541B%u5B89%u4E2D%u8BC11000%u4F18%u9009%u80A1%u7968%u53D1%u8D77A@%23%24019505; EMFUND8=07-23 20:30:12@#$%u56FD%u6CF0%u541B%u5B89%u4E2D%u8BC11000%u4F18%u9009%u80A1%u7968%u53D1%u8D77C@%23%24019506; websitepoptg_api_time=1722678907399; st_si=65977946119290; emshistory=%5B%22%E6%89%93%E6%96%B0%22%5D; HAList=ty-0-300059-%u4E1C%u65B9%u8D22%u5BCC%2Cty-1-603391-C%u529B%u805A%2Cty-1-603310-%u5DCD%u534E%u65B0%u6750%2Cty-0-301608-C%u535A%u5B9E%u7ED3%2Cty-0-301587-%u4E2D%u745E%u80A1%u4EFD%2Cty-1-000985-%u4E2D%u8BC1%u5168%u6307%2Cty-1-600961-%u682A%u51B6%u96C6%u56E2%2Cty-1-601899-%u7D2B%u91D1%u77FF%u4E1A%2Cty-0-000612-%u7126%u4F5C%u4E07%u65B9%2Cty-1-000852-%u4E2D%u8BC11000; st_asi=delete; st_pvi=78210881741202; st_sp=2023-11-30%2015%3A17%3A22; st_inirUrl=https%3A%2F%2Fwww.bing.com%2F; st_sn=145; st_psi=20240804112944896-117001354293-8593257916',
            'Origin': 'https://guba.eastmoney.com',
            'Referer': 'https://guba.eastmoney.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not)A;Brand";v="99", "Microsoft Edge";v="127", "Chromium";v="127"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            # 'Upgrade-Insecure-Requests': '1',
            # 'TE': 'Trailers',
            # 'Connection': 'keep-alive',
        }

url = 'https://guba.eastmoney.com/api/getData'


params = {
    'code' : 'cjpl',
    'path' : 'reply/api/Reply/ArticleNewReplyList'
}

cookies = 'qgqp_b_id=14677dc9de76f0637d3b2c0b49f08cc0; st_nvi=n8ARPwlVx_Ebz4lCQqxKm51b3; nid=090cd85fdab3bead76a0997310db6ffc; nid_create_time=1762149295492; gvi=arbT1iNmGcWnqifg6FlZwe12e; gvi_create_time=1762149295492; fullscreengg=1; fullscreengg2=1; st_si=65277115559484; st_asi=delete; st_pvi=43949517160482; st_sp=2025-11-03%2013%3A54%3A55; st_inirUrl=https%3A%2F%2Fwww.google.com.hk%2F; st_sn=3; st_psi=20251119181707746-117001354293-9234397977'
def cookie_str_to_dict(cookie_str):
    cookies = {}
    for item in cookie_str.split(';'):
        if '=' in item:
            k, v = item.strip().split('=', 1)
            cookies[k] = v
    return cookies
cookies_dict = cookie_str_to_dict(cookies)

# 1) 用 GET 发送 query（多数页面请求是这样）
payload = {
    'param': 'postid=1627355623&sort=1&sorttype=1&p=4&ps=30',
    'plat': 'Web',
    'path': 'reply/api/Reply/ArticleNewReplyList',
    'env': '2',
    'origin' : ' ',
    'version': '2022',
    'product': 'Guba'
}

session = requests.Session()

response = session.post(url, params=params, headers=headers, data=payload, cookies=cookies_dict)

data = response.json()

print(len(data['re']))

with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)