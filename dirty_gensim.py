documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

f = open('data.txt')
documents = f.readlines()
f.close()

# stoplist = set('for a of the and to in'.split())
# texts = [[word for word in document.lower().split() if word not in stoplist]
#          for document in documents]

from nltk.corpus import RegexpTokenizer
from nltk.corpus import stopwords

texts = []
for sentence in documents:
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    texts.append(filtered_words)

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

from pprint import pprint  # pretty-printer
from gensim import corpora
# pprint(texts)

dictionary = corpora.Dictionary(texts)
# print(dictionary)

# new_doc = "Computer AI"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

corpus = [dictionary.doc2bow(text) for text in texts]
# for c in corpus:
#     print(c)

from gensim import models
from gensim import similarities

##
##
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
##
##
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)

doc = 'What worries me about AI'#documents[0]
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
# print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus_tfidf]) # transform corpus to LSI space and index it

sims = index[vec_lsi] # perform a similarity query against the corpus
# for i, sim in enumerate(sims):
#     print('{} - {}'.format(sim, documents[i]))

sims_s = sorted(list(enumerate(sims)), key=lambda tup: tup[1], reverse=True)
for item in sims_s:
    i = item[0]
    v = item[1]
    print("{} - {}".format(v,documents[i]))



