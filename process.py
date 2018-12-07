#coding=utf-8
#计算五个文本之间的相似度，实现去除停用词功能，统计词频分布

from gensim import corpora, models, similarities
import jieba
from collections import defaultdict
from operator import itemgetter

DOC_PATH = ['./data/1霸王别姬.txt', './data/2罗马假日.txt', './data/3活着.txt', './data/4枪火.txt', './data/5无间道.txt',
            './data/6天堂电影院.txt', './data/7教父.txt', './data/8战狼二.txt']
DOC_NAME = ['霸王别姬', '罗马假日', '活着', '枪火', '无间道','天堂电影院', '教父', '战狼二']
NEW_DOC_PATH = ['./data/黑社会.txt']
STOPWORD_PATH = 'stopword.txt'
NEW_DOC_NAME = ['黑社会']


def delete_stopwords(dirty_texts, stopwords):
    '''
    去除停用词
    return: 列表clean_texts
    '''
    clean_texts = []
    for word in dirty_texts:
        if word not in stopwords:
            clean_texts.append(word)

    return clean_texts

def preprocess_doc(paths):
    '''
    读文档，并且分词
    return: 列表texts，存放单词
    '''
    stopwords = [line.strip() for line in open(STOPWORD_PATH).readlines()]

    docs = []
    for path in paths:
        doc = open(path).read()
        docs.append(doc)
        #print(type(doc))

    data = []
    for doc in docs:
        dirty_texts = jieba.cut(doc)
        clean_texts = delete_stopwords(dirty_texts, stopwords)
        data.append(clean_texts)

    documents = []
    for item in data:
        string = ""
        for word in item:
            string += word+" "
        documents.append(string)

    texts = [[word for word in document.split()]
        for document in documents
    ]
    return texts


def display(sim):
    result = []
    for s, n in zip(sim, DOC_NAME):
        result.append((s, n))

    result = sorted(result, key=itemgetter(0), reverse=True)

    for i in result:
        print("《%s》与《%s》的影评之间的相似度为: %.4f%%." % (NEW_DOC_NAME[0], i[1], i[0] * 100))


def main():
    texts = preprocess_doc(DOC_PATH)            #比较文档集
    new_text = preprocess_doc(NEW_DOC_PATH)     #新文档

    frequency = defaultdict(int)    #统计词频
    for text in texts:
        for word in text:
            frequency[word]+=1

    dictionary = corpora.Dictionary(texts)  #定义一个字典

    corpus_bow = [dictionary.doc2bow(text) for text in texts]       #词袋模型，得到每一篇文档的稀疏向量表示，向量的每一个元素代表了一个word在这篇文档中出现的次数。
    new_bow = dictionary.doc2bow(new_text[0])


    tfidf_model = models.TfidfModel(corpus_bow)       #tfidf模型，其中corpus是返回bow向量的迭代器，将完成对corpus中出现的每一个特征的IDF值的统计工作。


    new_tfidf = tfidf_model[corpus_bow]       #可以调用这个itidf_model将任意一段语料（依然是bow向量的迭代器）转化成TFIDF向量的迭代器
    new_vec_tfidf = tfidf_model[new_bow]

    featureNum = len(dictionary.token2id.keys())

    index = similarities.SparseMatrixSimilarity(new_tfidf, num_features=featureNum)
    sim = index[new_vec_tfidf]

    display(sim)

    return 0

if __name__ == '__main__':
    main()

