from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
from string import punctuation
import time
import re

start_time = time.time()

# Pandas function to load bugs from csv
bugs = pd.read_csv('./data/query_result_4097.csv')

# ref : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html
# Basically this will fill empty values with specified string in this case empty string
bugs.fillna('')
custom_stopwords = set(stopwords.words('english') + list(punctuation))
stemer = SnowballStemmer('english')

# This will clear subjects and results expected:
# original: Smart Banner (M) - Smart Banner either does not display or does not link correctly to Fox Now'
# cleared: Smart Banner M Smart Banner either display link correctly Fox Now

# function to filter stop words defined in custom_stopwords
def filter_stopwords(text):
    if not isinstance(text, str): return ''
    regexp = re.compile('(?u)\\b\\w\\w+\\b')
    regexped = ' '.join(regexp.findall(text))
    return ' '.join([word.lower() for word in word_tokenize(regexped) if word not in custom_stopwords])

def stem_map(text):
    return ' '.join([stemer.stem(word) for word in word_tokenize(text)])

bugs['tokenized_subject'] = list(map(filter_stopwords, bugs.subject))
bugs['tokenized_result'] = list(map(filter_stopwords, bugs.result))
bugs['tokenized'] = bugs.tokenized_subject + ' ' + bugs.tokenized_result

# Stemming will find and bring back word root
# example:
# stemmed: 
# 'smart banner m smart banner either display link correct fox now a staging.fox.com 1 safari no smart banner display way open episod fox now 2 chrome occasion smart banner display -but transfer fox now episod play time no smart banner display 3 firefox no smart banner display way open episod fox now b www.fox.com 1 safari same 2 chrome same 3 firefox same'
# original:
# 'Smart Banner M Smart Banner either display link correctly Fox Now A Staging.Fox.com 1 Safari No Smart Banner displays way open episode Fox Now 2 Chrome Occasionally Smart Banner displays -but transfer Fox Now episode play time No Smart Banner displays 3 Firefox No Smart Banner displays way open episode Fox Now B www.Fox.com 1 Safari Same 2 Chrome Same 3 Firefox Same'
bugs['stemmed'] = list(map(stem_map,bugs.tokenized))

bugs[['bugId', 'testerId', 'testCycleId', 'productId','lastRejectedReasonId', 'subject', 'tokenized', 'stemmed']].to_csv('./data/query_result_4097_cleared.csv')

elapsed_time = time.time() - start_time
print('Data cleaned in ', elapsed_time)
print('Results in ./data/query_result_4097_cleared.csv')