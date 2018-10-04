
from __future__ import division, absolute_import, print_function

import chapter1.chapter1 as c1

TITLE = '统计学习方法-李航'
INTRODUCTION = '统计学习是计算机及其应用领域的一门重要学科, \
系统地介绍了统计学习的主要方法,特别是监督学习方法,包括感知机, \
k近邻法,朴素贝叶斯法,决策树,Logistic回归,最大熵模型, \
支持向量机(SVM),提升方法,EM算法,隐马尔可夫模型,条件随机场'

def main():
    print(TITLE)
    print(INTRODUCTION)
    c1.main()
    
if __name__ == '__main__':
    main()
