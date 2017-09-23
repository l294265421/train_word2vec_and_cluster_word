# -*- encoding:utf-8 -*-

from sklearn.cluster import KMeans

baseDir = r'/home/liyuncong/program/dataset/news/sougo/cs/2012/'

with open(baseDir + "word2vec.vector") as word_vector_file:
    words = []
    vectors = []
    word_vector_file.readline()
    line = word_vector_file.readline()
    while line:
        tmp = line.split()
        words.append(tmp[0])
        vectors.append(tmp[1:])
        line = word_vector_file.readline()

    kmeans = KMeans(n_clusters=1000, n_jobs=-1)
    kmeans.fit(vectors)
    for index, cluster_id in enumerate(kmeans.labels_):
        with open(baseDir + 'word_class/' + str(cluster_id), mode= 'a') as words_file:
            words_file.write(words[index] + '\n')