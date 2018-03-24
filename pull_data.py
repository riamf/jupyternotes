import json
import requests as req

resp = req.get('https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty')

tag_list = json.loads(resp.content)
f = open('data.txt', 'w')
for tag in tag_list:
    resp = req.get('https://hacker-news.firebaseio.com/v0/item/{}.json?print=pretty'.format(tag))
    if resp.status_code == 200:
        json_object = json.loads(resp.content)
        title = json_object['title']
        f.write('{}\n'.format(title))
        print('{}'.format(title))

f.close()
