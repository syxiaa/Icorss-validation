'''
Author: Zhou Hao
Date: 2021-09-05 18:41:38
LastEditors: Zhou Hao
LastEditTime: 2021-10-28 21:19:40
Description: file content
E-mail: 2294776770@qq.com
'''
import numpy
import math

def Precision(y, y_hat):
    true_positive = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    Y = len(y_hat) - sum(y_hat)
    return true_positive / Y

def TPR(y, y_hat):
    true_positive = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_positive = len(y) - sum(y)
    return true_positive / actual_positive

def TNR(y, y_hat):
    true_negative = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_negative = sum(y)
    return true_negative / actual_negative

def get_Gmean(y, y_hat):
    tnr = TNR(y, y_hat)
    tpr = TPR(y, y_hat)
    return math.sqrt(tnr * tpr)

def get_Fmeature(y, y_hat, b=1):
    recall = TPR(y, y_hat)
    precision = Precision(y, y_hat)
    return ((1 + b * b) * recall * precision) / ((b * b) * recall + precision)


def accu():
    pass