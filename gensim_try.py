from collections import defaultdict
from gensim import corpora
from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer
from nltk.tokenize import word_tokenize
from pprint import pprint
from gensim.corpora import MmCorpus

f = open('data2.txt', 'r')
documents = f.readlines()
f.close()

texts = list(map(word_tokenize, documents))
pprint(texts[:3])

#remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for word in text:
        frequency[word] += 1

texts = [[word for word in text if frequency[word] > 1] for text in texts]
pprint(texts[:3])

# bag of words // word: (how many times it occures)
dictionary = corpora.Dictionary(texts)
# dictionary.save('./gen_sim.dict')

# tokenize to vectors
corpus = [dictionary.doc2bow(text) for text in texts]
pprint(corpus[:3])
# MmCorpus.serialize('./gen_sim_corpus.mm', corpus)


from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
tfidf = TfidfModel(corpus)
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])
corpus_tfidf = tfidf[corpus]
# for doc in corpus_tfidf:
#     print(doc)

lsi = LsiModel(corpus_tfidf, id2word=dictionary)
corpus_lsi = lsi[corpus_tfidf]
print(lsi.print_topics(2))

# for doc in corpus_lsi:
#     print(doc)

# lsi.save('gensim_lsi_model.lsi')

# transform corpus to LSI space and index it
index = MatrixSimilarity(lsi[corpus_tfidf])
# index.save('./gensim_indexes.index')

query = 'What worries me about AI'
query_vec = dictionary.doc2bow(query.lower().split())
# convert the query to LSI space
vec_lsi = lsi[query_vec]
print(vec_lsi)

# perform a similarity query against the corpus
sims = index[vec_lsi]
sims_s = sorted(list(enumerate(sims)), key=lambda tup: tup[1], reverse=True)
# sorted (document number, similarity score) 2-tuples
print('\n')
print('Printing first 10')
c = 0
for item in sims_s:
	i = item[0]
	v = item[1]
	c+=1
	if c==10: break
	print("{} - {}".format(v,documents[i]))



