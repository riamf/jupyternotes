import pandas as pd
import requests
import xml.etree.ElementTree as ET

response = requests.get("https://news.ycombinator.com/rss")
root = ET.fromstring(response.content)
channel = root.getchildren()[0]
items = channel.findall("item")
print("Found", len(items), "documents ")

documents = []
[documents.append(i.find("title").text) for i in items]
for doc in documents:
    print(doc)

from collections import Counter

doc_count = []
for doc in documents:
    c_tuple = Counter(doc.split()).most_common()
    doc_count.append(c_tuple)

for touples in doc_count:
    for t in touples:
        print('{} - {}'.format(t[0], t[1]))