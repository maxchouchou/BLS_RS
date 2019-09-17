BLS_RS
===================


这是一个基于[Borad Learning System](https://ieeexplore.ieee.org/document/7987745)方法(以下简称BLS)的电影推荐系统，也被称为宽度学习系统，是由澳门大学的陈俊龙教授在2017年TNNLS上基于随机向量函数链接神经网络(RVFLNN)和单层前馈神经网络(SLFN)提出的一种单层增量式神经网络。为了取得更高的拟合效率，BLS省略传统神经网络系统中的隐含层，通过在输入层中加入增强层的方式来进行非线性拟合，避免隐含层过多导致的梯度消失以及训练时间长等问题。本项目将BLS应用到推荐系统(Recommendation System)领域来改进先用的基于其他常见神经网络的机器学习方法，提高推荐系统的效率和准确率。


----------


安装和运行指南
-------------

以下指南将帮助你在本地机器上安装和运行该项目，进行开发和测试。

> - 使用软件：MATLAB 2015A
> - 数据来源：[MovieLens](https://movielens.org/) movie ml-1m dataset
> - 代码参考：[Broad Learning System](https://github.com/jash-git/Broad-Learning-System-MATLAB)
