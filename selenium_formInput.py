from selenium import webdriver
import time
from selenium.webdriver.support.ui import Select
import ddddocr
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions

options=ChromeOptions()
# options.add_argument('--headless') # 无头模式
# options.add_argument('--disable-javascript') # 禁用javascript


info={
'email':'123@qq.com',
'name':'张三',
'nickname':'阿三',
'tel':'13888888888',
'password':'nicaibudao666',
'company':'XXX有限公司',
'location':'河北|唐山',
'industy':'畜牧养殖'
}

url='http://www.goootech.com/signup_personal.aspx'

chrome=webdriver.Chrome(options=options)
chrome.get(url)
chrome.set_page_load_timeout(2)
time.sleep(1)

#获取各个控件
email=chrome.find_element(By.ID,'UEmail')
name=chrome.find_element(By.ID,'URealName')
nickname=chrome.find_element(By.ID,'UNickName')
tel=chrome.find_element(By.ID,'UTel')
password=chrome.find_element(By.ID,'UPwd1')
password_again=chrome.find_element(By.ID,'UPwd2')
company=chrome.find_element(By.ID,'UCompany')


province=Select(chrome.find_element(By.ID,'ChinaProvince'))
city=Select(chrome.find_element(By.ID,'ChinaCity'))

industy=chrome.find_element(By.XPATH,'//*[@id="UTrade"]/tbody/tr/td/label[text()="'+info['industy']+'"]')

caprcha_id=chrome.find_element(By.XPATH,"//input[@id='UValicode']")
caprcha_img=chrome.find_element(By.ID,'chkimg')

caprcha_img.screenshot('CaptchaImg/ckimg.png')


#填写信息
email.send_keys(info['email'])
time.sleep(0.5)
name.send_keys(info['name'])
time.sleep(0.5)
nickname.send_keys(info['nickname'])
time.sleep(0.5)
tel.send_keys(info['tel'])
time.sleep(0.5)
password.send_keys(info['password'])
time.sleep(0.5)
password_again.send_keys(info['password'])
time.sleep(0.5)
company.send_keys(info['company'])
time.sleep(0.5)

province.select_by_visible_text(info['location'].split('|')[0])
time.sleep(0.5)
city.select_by_visible_text(info['location'].split('|')[1])
time.sleep(0.5)
industy.click()
#验证码图片识别
ocr = ddddocr.DdddOcr()
with open('CaptchaImg/ckimg.png','rb') as f:
    img_bytes = f.read()
res = ocr.classification(img_bytes)
caprcha_id.send_keys(res)

time.sleep(20)
