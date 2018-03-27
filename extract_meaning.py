import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

def getArticle(url):
    page = requests.get(url).content.decode('utf8')
    soup = BeautifulSoup(page, 'lxml')
    text = ''.join(map(lambda p: p.text, soup.find_all('article')))
    text.encode('ascii', errors='replace')
    return text

def summarize(article, n):
    sents = sent_tokenize(text)

    assert n <= len(sents)

    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation) + ['”','“'])
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)

    tmp = nlargest(10, freq, key=freq.get)
    ranking = defaultdict(int)

    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]

    print(ranking)
    sents_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sents_idx)]

url = 'https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?utm_term=.7f2fa9c90c5d'
text = getArticle(url)


summary = summarize(text, 3)
print(summary)
