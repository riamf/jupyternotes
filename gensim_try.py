from collections import defaultdict
from gensim import corpora
from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer
from nltk.tokenize import word_tokenize
from pprint import pprint
from gensim.corpora import MmCorpus
import pandas as pd

# f = open('data2.txt', 'r')
# documents = f.readlines()
# f.close()

bugs = pd.read_csv('./data/query_result_4189_cleared.csv')
documents = bugs.tokenized

texts = list(map(word_tokenize, documents))

#remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for word in text:
        frequency[word] += 1

texts = [[word for word in text if frequency[word] > 1] for text in texts]

# bag of words // word: (how many times it occures)
dictionary = corpora.Dictionary(texts)
dictionary.save('./gen_sim.dict')

# tokenize to vectors
corpus = [dictionary.doc2bow(text) for text in texts]
# MmCorpus.serialize('./gen_sim_corpus.mm', corpus)


from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity


lsi = LsiModel(corpus, id2word=dictionary)
corpus_lsi = lsi[corpus]

lsi.save('gensim_lsi_model.lsi')

# transform corpus to LSI space and index it
index = MatrixSimilarity(corpus_lsi)
index.save('./gensim_lsi_matrix_similarity.index')

query = documents[0]
query_vec = dictionary.doc2bow(query.lower().split())
# convert the query to LSI space
vec_lsi = lsi[query_vec]

# perform a similarity query against the corpus
sims = index[vec_lsi]
sims_s = sorted(list(enumerate(sims)), key=lambda tup: tup[1], reverse=True)
# sorted (document number, similarity score) 2-tuples
print('\n')
print('Printing first 10')
real_documents = bugs.subject
c = 0
for item in sims_s:
	i = item[0]
	v = item[1]
	c+=1
	if c==10: break
	print("{} - {}".format(v,real_documents[i]))



