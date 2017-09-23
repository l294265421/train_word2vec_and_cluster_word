# -*- coding: utf-8 -*-

from gensim.models import word2vec

baseDir = r'/home/liyuncong/program/dataset/news/sougo/cs/2012/'

# 加载语料
# utf8_news_sohusite_xml.englishlike.txt是通过搜狗语料库生成的文件，其中，词已经通过空格分开
# 原始搜狗数据地址为http://www.sogou.com/labs/resource/cs.php，完整版
# utf8_news_sohusite_xml.englishlike.txt地址为http://pan.baidu.com/s/1c19mgEG
sentences = word2vec.Text8Corpus(baseDir + u"utf8_news_sohusite_xml.englishlike.txt")
# 训练skip-gram模型; 默认window=5
model = word2vec.Word2Vec(sentences, size=200)

# 保存模型
model.save(baseDir + u"word2vec.model")
# 对应的加载方式
model_2 = word2vec.Word2Vec.load(baseDir + u"word2vec.model")

# 以一种C语言可以解析的形式存储词向量
# model.save_word2vec_format(u"书评.model.bin", binary=True)
model.wv.save_word2vec_format(baseDir + u"word2vec.model")
# 对应的加载方式
model_3 = word2vec.Word2Vec.load_word2vec_format(baseDir + u"word2vec.model", binary=True)

if __name__ == "__main__":
    pass