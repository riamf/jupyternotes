import gensim
from nltk.tokenize import word_tokenize

f = open('data.txt', 'r')
raw_documents = f.readlines()
f.close()

gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]


dictionary = gensim.corpora.Dictionary(gen_docs)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print('items in corpus: ', len(corpus))

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

sims = gensim.similarities.Similarity('./',tf_idf[corpus],num_features=len(dictionary))

query_doc = ['mobile', 'android', 'ios']
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

similarities = sims[query_doc_tf_idf]

min_sim = 0.1
print('query: ', query_doc)
for i, doc in enumerate(raw_documents):
    val = similarities[i]
    if val > min_sim: print('{} - {} - {}'.format(i,val,doc)) 



