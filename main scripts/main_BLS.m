% main_BLS.m
tic

%% 基于宽度学习推荐
%% 说明
% 1、功能：使用宽度学习方法进行推荐。
% 2、运行时间：2.5秒(10000行数据)
% 3、运行需要的输入：ml-1m-train.mat
% 4、运行输出：ml-1m-bls.mat
% 5、代码关联：main_Datacreate.m，无后序代码

%% I.初始化环境
%%
clear;
clc;
warning off;
fprintf('I.初始化环境...\n');
fprintf('\n');

%% II.数据导入
%%
fprintf('II.数据导入...\n');
load ml-1m-train.mat;
fprintf('\n');

% 提取前10000行运算
datanum=10000;
trainnum=round(datanum*0.7);
p_train=p_train(:,1:trainnum);
P_train=P_train(:,1:trainnum);
t_train=t_train(:,1:trainnum);
T_train=T_train(:,1:trainnum);
testnum=datanum-trainnum;
p_test=p_test(:,trainnum+1:datanum);
P_test=P_test(:,trainnum+1:datanum);
T_test=T_test(:,trainnum+1:datanum);

% 存入模型
bls.data.train_x=p_train';
bls.data.train_y=t_train';
bls.data.test_x=p_test';
bls.data.test_y=T_test';

%% III.网络创建、训练及仿真测试
%%
fprintf('III.网络创建、训练及仿真测试...\n');

% 1.模型结构/参数
fprintf('参数设置...\n');
bls.parameter.C = 2^-30;% l2正则化参数 系数正则化（自编码）的正则化参数
bls.parameter.s = 0.8; % 增强节点的收缩尺度
bls.parameter.N1=15; % 每个窗口的特征节点
bls.parameter.N2=3; % 窗口数
bls.parameter.N3=200; % 增强节点数
bls.parameter.epochs=1 ; % 迭代次数
rand('state',1);
fprintf('\n')

% 2.训练
fprintf('训练...\n');
[bls] = bls_train(bls); 
fprintf('\n')

% 3.测试
fprintf('测试...\n');
[bls] = bls_test(bls); 
fprintf('\n')

% 4.反归一化
bls.test.pre=mapminmax('reverse',bls.test.output{1,1},ps_output);
fprintf('\n');

%% IV.性能评价
%%
fprintf('IV.性能评价...\n');
pre=bls.test.pre;
true=bls.data.test_y;
test_x=bls.data.test_x;

% 1. 计算RMSE
bls.evaluate.RMSE=sqrt(sum((pre-true).^2)/(testnum));

% 2. 计算决定系数R^2
[~,~,~,~,stats] = regress(pre, true);
bls.evaluate.R2 = stats(1);

% 3. Sparsity
bls.evaluate.sparsity=sum(sum(pre==0)/(testnum));

% 4. accuracy
meanpre=bsxfun(@minus, pre,P_test(5,:)');
meantrue=bsxfun(@minus, true,P_test(5,:)');
bls.evaluate.accuracy=sum(sum(sign(meanpre)==sign(meantrue))/(testnum));

fprintf('\n');

clearvars -except bls;

%% V.存储模型
%% 
save('ml-1m-bls.mat');

toc
