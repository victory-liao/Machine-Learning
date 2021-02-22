# coding:utf-8
import operator
import numpy as np
from os import listdir

def img2vector(filename):
    vector = np.zeros((1, 1024))

    f = open(filename)

    for i in range(32):
        line = f.readline()
        for j in range(32):
            vector[0, 32*i+j] = line[j]

    return vector

def classify(test_data, training_data, traing_lables, k):
    # 获取训练数据矩阵的行
    training_data_size = training_data.shape[0]
    # 测试矩阵-训练矩阵
    sub_mat = np.tile(test_data, (training_data_size, 1)) - training_data
    # 相减后平方
    sq_mat = sub_mat**2
    # 求和
    add_mat = sq_mat.sum(axis=1)
    # 开方得出距离
    distances = add_mat**0.5
    # 距离从小到大排序
    sorted_dist_indices = distances.argsort()

    class_count = { }
    # 取出前k小个距离对应元素的类别
    for i in range(k):
        label = traing_lables[sorted_dist_indices[i]]
        # 计算各类别出现的频率
        class_count[label] = class_count.get(label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sorted_class_count[0][0]

def digits_recognition_test():
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_data = np.zeros((m, 1024))
    training_data_labels = []
    for i in range(m):
        file_class = training_file_list[i].split('_')[0]
        training_data_labels.append(file_class)
        vec = img2vector('trainingDigits/%s' % training_file_list[i])
        training_data[i, :] = vec[:]

    test_file_list = listdir('testDigits')
    m_test = len(test_file_list)
    error_count = 0.0
    for i in range(m_test):
        test_file_data = img2vector('testDigits/%s' % test_file_list[i])
        test_file_class = test_file_list[i].split('_')[0]
        predicted_class = classify(test_file_data, training_data, training_data_labels, 3)
        print("真实分类:%s 预测分类:%s" % (test_file_class, predicted_class))
        if test_file_class != predicted_class:
            error_count += 1.0
    print("错误次数:%d 错误率:%f%%" % (error_count, error_count/m_test))


if __name__ == '__main__':
    digits_recognition_test()
    # vec1 = img2vector('trainingDigits/0_0.txt')
    # print(vec1)
