# 使用theano作为backend
#产生所有的信息组合，在[0, q-1],不在[-1, q-2]

import os
os.environ['KERAS_BACKEND']='theano'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # keras实现多GPU或指定GPU的使用 https://blog.csdn.net/qq_36427732/article/details/79017835

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model#导入网络结构可视化模块。 https://blog.csdn.net/baymax_007/article/details/83152108
import random
from random import randint



#Parameters
k = 3                       # number of information symbols
N = 15                     # code length
system = 16                # the number system



#define a eight system add fuction, 1 is added every time
def eight_add(aa):
    n_list = [1];
    m_list = list(aa);
    n_list = list(n_list);
    m_list.reverse()  # 翻转，方便从个位开始加
    n_list.reverse()
    # 加数和被加数补齐，防止数组越界，短者高位补0
    if len(m_list) > len(n_list):
        result = [''] * (len(m_list) + 1)  # 保存和，多一位是防止最高位也有进一的情况
        n_list = n_list + [0] * (len(m_list) - len(n_list))
    else:
        result = [''] * (len(n_list) + 1)
        m_list = m_list + [0] * (len(n_list) - len(m_list))

    flag = False
    for i in range(max(len(m_list), len(n_list))):
        if flag:  # 如果上一位有进1，本位和需要加上上一位进的1
            plus = int(n_list[i]) + int(m_list[i]) + 1
        else:
            plus = int(n_list[i]) + int(m_list[i])

        if plus >= system:  # 本位大于8，本位存本位和-8，并向前进一 ；if plus >= (system-1):
            result[i] = str(plus - system)#本来是system，比如8进制就是8，但是这里用(system-1)，主要是为了配合[-1, q-2]   result[i] = str(plus - (system-1))
            flag = True
        else:
            result[i] = str(plus)
            flag = False
    if flag:  # 最高位最终向前进1，和也需要向前进1
        result[-1] = str(1)  # python中数组下标为-1时 https://blog.csdn.net/jiayizhenzhenyijia/article/details/97623762
        # result的下标从0到5，最后一个元素是result[5]，但最后一个元素也可以用result[-1]来表示
    result.reverse()
    del (result[0]);  # 删除掉最高位元素，一个是因为不需要进位的那值，二个是一是因为如果没有进位，就是空值，下面的list(map(int,result))类型转换就会出问题。
    int_result = list(map(int, result))  # 字符型转化为int型， https://blog.csdn.net/tutu96177/article/details/87
    return int_result


# Create all possible information words
d = np.zeros((system ** k, k), dtype=int)
# d = d-1 #不在[0, q-1],而在[-1, q-2]上，和matlab里通过唯一好用rs码产生的MDS码字相对应，那个码字就是在[-1, q-2],素以-1
for i in range(1, system ** k):
    d[i] = eight_add(d[i-1])

np.savetxt("dets.txt", d, fmt='%d', delimiter=',')#将所有信息存储成txt文件，放到matlab里用rs码编码得到所有码字https://blog.csdn.net/qq_38497266/article/details/88871197

