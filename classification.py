

f = open('data.txt', 'r')
documents = f.readlines()
f.close()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=2,
    stop_words='english'
)
X = vectorizer.fit_transform(documents)
print(X)
n_clusters = 3
km = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    max_iter=100,
    n_init=1,
    verbose=True
)
km.fit(X)

tmp = np.unique(km.labels_, return_counts=True)
print(tmp)
text = {}
for i,cluster in enumerate(km.labels_):
    one_document = documents[i]
    if cluster not in text.keys():
        text[cluster] = one_document
    else:
        text[cluster] += one_document

for value,key in text:
    print(key)