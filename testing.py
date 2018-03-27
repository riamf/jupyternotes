from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.stem import LancasterStemmer

f = open('data.txt', 'r')
lines = f.readlines()
f.close()

custom_stopwords = set(stopwords.words('english') + list(punctuation))

tokenized_lines = []
for line in lines:
    tokenized_words = [word for word in word_tokenize(line) if word not in custom_stopwords]
    tokenized_lines.append(tokenized_words)

bigram_measures = BigramAssocMeasures()
ngrams = []
for line in tokenized_lines:
    ngrams.append(sorted(BigramCollocationFinder.from_words(line).ngram_fd.items()))


st = LancasterStemmer()
stemmed = []
for line in tokenized_lines:
    stemmed_words = [st.stem(word) for word in line]
    stemmed.append(stemmed_words)

for st in stemmed:
    print(st)