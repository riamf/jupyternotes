from gensim.similarities import MatrixSimilarity
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import pandas as pd
import time

bugs = pd.read_csv('./data/query_result_4189_cleared.csv')
documents = bugs.tokenized
real_documents = bugs.subject

lsi = LsiModel.load('gensim_lsi_model.lsi')
dictionary = Dictionary.load('./gen_sim.dict')
index = MatrixSimilarity.load('./gensim_lsi_matrix_similarity.index')

start = time.time()

query = documents[0]
query_vec = dictionary.doc2bow(query.lower().split())
# convert the query to LSI space
vec_lsi = lsi[query_vec]

sims = index[vec_lsi]
sims_s = sorted(list(enumerate(sims)), key=lambda tup: tup[1], reverse=True)

end = time.time()
print('Counting one query took: {}'.format(end - start))
c = 0

for item in sims_s:
	i = item[0]
	v = item[1]
	c+=1
	if c==10: break
	print("{} - {}".format(v,real_documents[i]))


def top_similarities(query,threshold=0.5, max_count=10):
    query_vec = dictionary.doc2bow(query.lower().split())
    vec_lsi = lsi[query_vec]
    sims = index[vec_lsi]
    sims_s = sorted(list(enumerate(sims)), key=lambda tup: tup[1], reverse=True)
    c = 0
    results = []
    for item in sims_s:
        i = item[0]
        v = item[1]
        c+=1
        if c==max_count or v < threshold: break
        results.append((c,v,real_documents[i]))
    return results

all_sims = []
f = open('res.txt','w')
c = 0
l = len(documents)
for doc in documents:
    results = top_similarities(doc)
    all_sims.append(results)
    c+=1
    print('{}/{} 👍'.format(c,l))
    results_s = '\n'.join(['{} - {}'.format(r[1],r[2]) for r in results])
    result_s = '{}\n{}\n'.format(doc,results_s)
    f.write(result_s)

f.close()