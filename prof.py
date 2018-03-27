import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

f = open('data.txt', 'r')
lines = f.readlines()
f.close()

print('Lines loaded')
stemmer = SnowballStemmer('english')
start = time.time()
stemmed = [stemmer.stem(line) for line in lines]
end = time.time()
print('Stemmed in {}s'.format(start - end))

cvec = CountVectorizer(stop_words='english', min_df=0.0075, max_df=.9, ngram_range=(1,2))
from itertools import islice
cvec.fit(stemmed)
tmp = list(islice(cvec.vocabulary_.items(), 20))
print(tmp)
print(len(cvec.vocabulary_))


cvec_counts = cvec.transform(stemmed)
print('sparse matrix shape:', cvec_counts.shape)
print('nonzero count:', cvec_counts.nnz)
print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))

occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
tmp = counts_df.sort_values(by='occurrences', ascending=False).head(20)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
tmp = weights_df.sort_values(by='weight', ascending=False).head(20)
print(tmp)