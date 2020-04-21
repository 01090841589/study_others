from bs4 import BeautifulSoup
import urllib.request
import requests
from urllib.parse import quote_plus
import pickle
# url = "https://www.netcarshow.com/"


car_lists = set()
for i in range(1, 7):
    url = "https://auto.naver.com/car/mainList.nhn?importYn=N&page="+str(i)

    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'html.parser')

    lists = soup.find('div', class_='model_group_new')
    label = lists.find_all('span', {'class':'box'})
    label = lists.find_all('strong')
    for lab in label:
        car_lists.add(lab.text)

# for i in range(1, 88):
#     url = "https://auto.naver.com/car/mainList.nhn?importYn=Y&page="+str(i)

#     res = requests.get(url)

#     soup = BeautifulSoup(res.content, 'html.parser')

#     lists = soup.find('div', class_='model_group_new')
#     label = lists.find_all('span', {'class':'box'})
#     label = lists.find_all('strong')
#     for lab in label:
#         car_lists.add(lab.text)
# print(len(car_lists))

# save
with open('data.pickle', 'wb') as f:
    pickle.dump(car_lists, f, pickle.HIGHEST_PROTOCOL)