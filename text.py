# coding: utf-8

import sys
import jieba  # jieba pyhon库 支持分词等操作
import numpy  # numpy pyton库 用于矩阵运算
numpy.set_printoptions(threshold=100000)  # 设置numpy 打印长度

from sklearn import metrics  # sklearn python库 提供机器学习的函数
from sklearn.feature_extraction.text import HashingVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split  # 分割数据集

# 归一化处理
from sklearn.preprocessing import MinMaxScaler


def data_split(x, y):  # x  word y tag
    word_train = []
    tag_train = []
    word_test = []
    tag_test = []
    # 测试集为10%，训练集为90%
    with open(x, 'r') as f1:
        for line in f1:  # 获取训练集数据
            line = line[:-1]
            word_train.append(line)
    with open(y, 'r') as f1:
        for line in f1:  # 获取测试集数据
            line = line[:-1]
            tag_train.append(line)
    with open('text1.txt','r') as f1:
        for line in f1:
            line = line[:-1]
            word_test.append(line)
    with open('text2.txt','r') as f1:
        for line in f1:
            line = line[:-1]
            tag_test.append(line)

    #x_train, x_test, y_train, y_test = train_test_split(word, tag, test_size=0.1, random_state=0)
    '''x_train是90%用户提问，y_train是90%标准答案,x_test是10%测试用的客户提问，y_test是10%标准答案'''
    '''
    for lines in x_train:
        print(lines)
    print("\n\n\n\n\n\n\n\n")
    for lines in x_test:
        print(lines)
    print("\n\n\n\n\n\n\n\n")
    for lines in y_train:
        print(lines)
    print("\n\n\n\n\n\n\n\n")
    for lines in x_test:
        print(lines)
    print("\n\n\n\n\n\n\n\n")'''
    return word_train,tag_train,word_test,tag_test  #x word  y tag


with open('hlt_stop_words.txt', 'r', encoding='UTF-8') as f:
    stopwords = set([w.strip() for w in f])  # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。 读取停用词


with open('Qword.txt', 'r', encoding='UTF-8') as f:
    qword = set([w.strip() for w in f])


def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords or word in qword:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


def jieba_tokenizer(x):  # 不去停用词效果更好
    return jieba.cut(x)  # jieba分词  全模式


def vectorize(train_words, test_words):  # 向量化函数 测试集和训练集只有共用一个vectorizer才能共享vocabulary，避免特征表达不一致的问题。
    # v = HashingVectorizer(tokenizer=jieba_tokenizer, n_features= 1024 , non_negative=True)
    # v = TfidfVectorizer(tokenizer=seg_sentence)
    # v = CountVectorizer(tokenizer=seg_sentence)
    v = HashingVectorizer(tokenizer=seg_sentence, n_features=800, non_negative=True)
    train_data = v.fit_transform(train_words)  #.todense
    test_data = v.fit_transform(test_words)
    return train_data, test_data


# 归一化处理
def normalization_processing(my_matrix):
    scaler = MinMaxScaler()
    scaler.fit(my_matrix)
    #scaler.data_max_
    my_matrix_normorlize = scaler.transform(my_matrix)
    return my_matrix_normorlize


def evaluate(actual, pred):  # 评测
    m_precision = metrics.precision_score(actual, pred, average="micro")
    #m_recall = metrics.recall_score(actual, pred, average="micro")

    print('正确率:{0:.6f}'.format(m_precision))
    #print('recall:{0:0.6f}'.format(m_recall))


def train_clf(train_data, train_tags):  # 分类器classifier
    clf = MultinomialNB(alpha=0.002)
    clf.fit(train_data, numpy.asarray(train_tags))  # 用训练集训练分类器
    return clf


def main():
    words_file = 'words_file.txt'  # 语句文件
    tag_file = 'tag_file.txt'  # 标签文件

    train_words, train_tags, test_words, test_tags = data_split(words_file, tag_file)
    # 语句和标签分割为测试集和训练集  从文件中获取数据

    train_data, test_data = vectorize(train_words, test_words)  # 语句向量化

    clf = train_clf(train_data, train_tags)  # 用训练集训练分类器
    sim0 = clf.predict_proba(test_data)  # 对于测试集中的语句，每一个语句对应一个矩阵，矩阵中是这个语句对应所有标签的可能性
    sim = clf.predict_log_proba(test_data)  # log对数后的sim0

    pred = clf.predict(test_data)  # 预测

    count = 0  # 计数器
    count2 = 0

    #result = normalization_processing(sim0)  # 对数组长为测试集大小的 矩阵的数组进行归一化处理
    result2 = normalization_processing(sim)
    #print(result2)

    for element in result2:  # 以归一化后的概率作为相似度
        #print(max(element))
        if max(element) > 0.8:
            count2 += 1

    '''
    for element in pred:  # 输出标准问
        print(element)
        
    print("\n\n\n\n\n\n\n\n")
    
    for element in test_words:  # 用户提问
        print(element)
    print("\n\n\n\n\n\n\n\n")
    
    for element in test_tags:# 正确标准问
        print(element)
    '''

    #print(count)
    for (line1,line2) in zip(pred,numpy.asarray(test_tags)):
        if(line1==line2):
            count = count+1

    pred = pred.reshape(pred.size, 1)
    test_tags = numpy.array(test_tags)
    test_tags = test_tags.reshape(test_tags.size, 1)
    a = numpy.hstack((pred, test_tags))
    print(a)

    n = a.shape[0]

    for i in range(n):
        print(a[i][0],a[i][1])

    print("测试集正确的个数:"+str(count))
    print("测试集总个数:"+str(count2))

    evaluate(numpy.asarray(test_tags), pred)
    print('召回率:{0:.6f}'.format(count/count2))


if __name__ == '__main__':
    main()

