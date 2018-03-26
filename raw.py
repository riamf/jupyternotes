import math

class TFIDF:

    def __init__(self, query, documents, lowercase = True):
        self.query = self.parse([query], lowercase)[0]
        self.original_documents = documents
        self.documents = self.parse(documents, lowercase)
        self.document = ' '.join(documents)
        self.word_bag = list(set(self.document.split()))
        self.D = len(documents)

    def parse(self, documents, lowercase):
        result = []
        for doc in documents:
            alphadoc = ''.join(ch for ch in doc if ch.isalpha() | ch.isspace())
            result.append(alphadoc.lower() if lowercase else doc)
        return result

    def TF(self, word, doc):
        return doc.split().count(word)
    
    def IDF(self, word):
        result = [sum(doc.count(word) for doc in self.documents)][0] + 1
        return math.log(self.D/result)


    def count(self):
        summaries = [0] * len(self.documents)
        for word in self.query.split():
            idf = self.IDF(word)
            # print("{}-{}".format(word,idf))
            w_ = []
            for (idx, doc) in enumerate(self.documents):
                tf = self.TF(word, doc)
                w = tf * idf
                # print("{} - {} - {} - {}".format(word,tf,idf,w))
                w_.append(w)
                summaries[idx]+=w
        result = list(zip(summaries, self.documents))
        return sorted(result, key=lambda tup: tup[0], reverse=True)

f = open('data.txt', 'r')
documents = f.readlines()
f.close()

tfidf = TFIDF('facebook data integration', documents)
result = tfidf.count()
for item in result:
    print(item)