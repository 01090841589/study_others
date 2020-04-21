import requests, json

data={
    'nickname' : "구미1반박찬환",
    'yourAnswer' : "(정답입력)"
    }

r = requests.get('http://ssafy-friends.com/',
    headers = {'Content-Type': 'application/json; charset=utf-8'},
    data=json.dumps(data)
)
print(r.json())
