from bs4 import BeautifulSoup
import requests
import pickle
import os
import time
import json
import urllib
from urllib.request import urlretrieve
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

with open('data.pickle', 'rb') as f:
    car_lists = pickle.load(f)

for i in car_lists:
    i = i.rstrip()
    fdir = os.path.dirname(os.path.abspath(__file__))+'/models'+'/'+i[:25]
    print(fdir)
    if os.path.exists(fdir) == False:
        os.mkdir(fdir)

    # img = "https://www.google.com/search?q="+i+"&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiB6d_Yj_boAhUEA4gKHUCkCQAQ_AUoAXoECAwQAw&biw=1396&bih=657"
    url = "https://search.naver.com/search.naver?where=image&sm=tab_jum&query="+i
    
    # url = "https://auto.naver.com/car/mainList.nhn?importYn=N&page=1"
    res = requests.get(url)    
    soup = BeautifulSoup(res.content, 'html.parser')

    lists = soup.find('div', class_='photowall _photoGridWrapper')
    image = lists.find_all('a', {'class':'thumb _thumb'})
    print(len(image))
    for j in range(min(len(image), 20)):
        one_image = image[j].find('img')
        one_image = str(one_image)
        one_image = one_image[one_image.index('data-source')+13:one_image.index('data-width')-2]
        print(j, "_naver : ", one_image)  
        urlretrieve(one_image, fdir+"/naver_{}.jpg".format(j))
    time.sleep(0.5)
    





options = webdriver.ChromeOptions()
options.add_argument('headless')
# google_url = "https://www.google.com/search?q="+i+"&tbm=isch&ved=2ahUKEwiPkqOHo_joAhVM15QKHSFiC8YQ2-cCegQIABAA&oq=%EC%95%84%EC%9A%B0%EB%94%94&gs_lcp=CgNpbWcQAzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIADICCAA6BQgAEIMBUIgOWOsRYNkSaAFwAHgCgAGCAYgBngWSAQMxLjWYAQCgAQGqAQtnd3Mtd2l6LWltZw&sclient=img&ei=yz-eXs-ABsyu0wShxK2wDA&bih=472&biw=766"
browser = webdriver.Chrome('../chrom/chromedriver.exe', options=options)

for i in car_lists:


    i = i.rstrip()
    fdir = os.path.dirname(os.path.abspath(__file__))+'/models'+'/'+i[:25]
    print(fdir)
    if os.path.exists(fdir) == False:
        os.mkdir(fdir)

    google_url = "https://www.google.co.in/search?q="+i+"&source=lnms&tbm=isch"
    browser.get(google_url)
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

    counter = 0
    succounter = 0
    time.sleep(0.5)
    
    html = browser.page_source
    
    time.sleep(0.5)


    soup = BeautifulSoup(html,'html.parser')
    img = soup.findAll("img")

    browser.find_elements_by_tag_name('img')

    fileNum=0

    time.sleep(0.5)
    imgnum = 1
    for line in img:
        if str(line).find('data-src') != -1 and str(line).find('http')<100:  
            print(fileNum, "_google : ", line['data-src'])  
            
            urlretrieve(line['data-src'], fdir+"/google_{}.jpg".format(fileNum))

            fileNum+=1
            if imgnum > 20:
                break
            imgnum += 1
        time.sleep(0.2)