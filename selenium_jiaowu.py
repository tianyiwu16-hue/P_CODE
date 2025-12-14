import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from ddddocr import DdddOcr

# Handle the import error gracefully

from selenium.webdriver import ChromeOptions

options=ChromeOptions()
options.add_argument('--referer=http://121.193.151.131/jwweb/home.aspx')
options.add_argument('--User-Agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"')
options.add_argument('--Accept="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"')
url='http://121.193.151.131/jwweb/home.aspx'

sno='2024111111'
pwd='1234'

# Fix Chrome service initialization
service = Service()  # Create service instance correctly
chrome=webdriver.Chrome(service=service, options=options)
chrome.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                            Object.defineProperty(navigator, 'webdriver', {
                              get: () => undefined
                            })
                          """
        })#解决网站对selenium架构的反扒限制

# chrome.get(url=url)
# chrome.switch_to.frame('frm_login')

# tb_Sno=chrome.find_element(By.ID,'txt_asmcdefsddsd')
# tb_Sno.send_keys(sno)
# time.sleep(5)
# #Method 1 通过键盘TAB切换激活隐藏控件
# # tb_Sno.send_keys(Keys.TAB)

# #Method 2 点击控件激活隐藏控件
# # txt_psasas=chrome.find_element_by_id('txt_psasas')
# # txt_psasas.click()

# #Method 3 通过执行js脚本改变密码框的显示样式
# chrome.execute_script("document.getElementById('txt_pewerwedsdfsdff').style.display='';")
# time.sleep(5)



# tb_Pwd=chrome.find_element(By.ID,'txt_pewerwedsdfsdff')
# tb_Pwd.send_keys(pwd)
# time.sleep(5)

# tb_captcha=chrome.find_element(By.ID,'txt_sdertfgsadscxcadsads')
# tb_captcha.click()
# time.sleep(1)
# img_captcha=chrome.find_element(By.ID,'imgCode')
# img_captcha.screenshot('CaptchaImg/jiaowu.png')

# ocr=DdddOcr()
# with open('CaptchaImg/jiaowu.png','rb') as f:
#     img=f.read()

# code=ocr.classification(img)
# tb_captcha.send_keys(code)

# time.sleep(15)
# #out
# # driver.get_cookies()

# cookies=chrome.get_cookies()
# print(chrome.get_cookies())
# time.sleep(10)

# with open("cookies.json", "w", encoding="utf-8") as cks:  # 把cookies使用json保存
#     json.dump(cookies, cks)



# #---------------读取-------------------------------
# # for i in get_cookies(): # 添加的核心，已经保存的cookies是个list，其中的才是cookie，使用for循环添加
# #         browser.add_cookie(i)
# # time.sleep(30)
with open("cookies.json", "r", encoding="utf-8") as cks:  # 把cookies使用json保存
    cookies=json.loads(cks.read())
url='http://121.193.151.131/jwweb/MAINFRM.aspx'
chrome.get(url)
chrome.add_cookie(cookies[0])
chrome.add_cookie(cookies[1])
chrome.add_cookie(cookies[2])
chrome.get(url)
time.sleep(600)