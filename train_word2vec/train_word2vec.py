# -*- coding: utf-8 -*-

from gensim.models import word2vec

baseDir = r'/home/liyuncong/program/dataset/news/sougo/cs/2012/'

# 加载语料
sentences = word2vec.Text8Corpus(baseDir + u"utf8_news_sohusite_xml.englishlike.txt")
# 训练skip-gram模型; 默认window=5
model = word2vec.Word2Vec(sentences, size=200)

# 保存模型
model.save(baseDir + u"word2vec.model")
# 对应的加载方式
model_2 = word2vec.Word2Vec.load("text8.model")

# 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
model.wv.save_word2vec_format(baseDir + u'word2vec.vector')
# 对应的加载方式
model_3 = word2vec.Word2Vec.load_word2vec_format("text8.model.bin", binary=True)

if __name__ == "__main__":
    pass