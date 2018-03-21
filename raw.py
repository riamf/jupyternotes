import pandas as pd
import requests
import xml.etree.ElementTree as ET

response = requests.get("https://news.ycombinator.com/rss")
root = ET.fromstring(response.content)
channel = root.getchildren()[0]
items = channel.findall("item")
print("Found", len(items), "documents ")

documents = []
for i in items:
    text = i.find("title").text
    res = ''.join(e for e in text if (e.isalnum() | e.isspace()))
    documents.append(res)

#joining words
document = ' '.join(documents)

for doc in documents:
    print(doc)

from collections import Counter

fwd = []
for doc in documents:
    c_tuple = Counter(doc.split()).most_common()
    fwd.append(c_tuple)

for touples in fwd:
    for t in touples:
        print('{} - {}'.format(t[0], t[1]))

D = len(document.split()) #Counting number of words
print('number of words: {}'.format(D))


# Counting f(w,D)



