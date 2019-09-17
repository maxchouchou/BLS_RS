BLS_RS
===================

这是一个基于[Borad Learning System](https://ieeexplore.ieee.org/document/7987745)方法(以下简称BLS)的电影推荐系统，也被称为宽度学习系统，是由澳门大学的陈俊龙教授在2017年TNNLS上基于随机向量函数链接神经网络(RVFLNN)和单层前馈神经网络(SLFN)提出的一种单层增量式神经网络。为了取得更高的拟合效率，BLS省略传统神经网络系统中的隐含层，通过在输入层中加入增强层的方式来进行非线性拟合，避免隐含层过多导致的梯度消失以及训练时间长等问题。本项目将BLS应用到推荐系统(Recommendation System)领域来改进先用的基于其他常见神经网络的机器学习方法，提高推荐系统的效率和准确率。

----------
以下指南将帮助你在本地机器上安装和运行该项目，进行开发和测试。


运行软件
-------------
> **基本情况**
> - 使用软件：MATLAB 2015A
> - 数据来源：[MovieLens](https://movielens.org/) movie ml-1m dataset
> - 代码参考：[Broad Learning System](https://github.com/jash-git/Broad-Learning-System-MATLAB)


文件夹列表
-------------
> - original data：从MovieLens网站上下载的原始数据文件，格式为dat。
> - processed data：根据.dat文件处理后的.xlsx文件，包括对数据的初步预处理，将电影的特征转化为布尔型等。
> - main scripts：基于MATLAB创建的函数和主函数。
> - outputoutput：输出结果，为MATLAB .mat格式文件。

运行指南
-------------
> - 原始和初步处理数据：可点开文件“original data”或“processed data”，包括电影特征数据“movies”，评分数据“ratings”，用户数据“user”以及原始数据创建者的README文件.

> - 开始运行项目通过运行main scripts中的主程序脚本。脚本命名格式为“main_”的为主程序文件，带有“bls_”为宽度学习函数，带有“rs_”的为推荐系统函数，带有“data_”的为数据文件。

> - 运行顺序：首先运行**main_CF**得到协同过滤推荐结果，其次运行**main_Datacreate**将得到的协同过滤结果转化为推荐输入。然后基于BLS、BLSBP(反向传播算法改进的BLS方法)、BPNN(传统的反向传播神经网络)分别对推荐输入进行模型构建和求解，得到最终结果。即运行**main_BLS**、**main_BLSBP**、**main_BPNN**对推荐系统进行求解。
