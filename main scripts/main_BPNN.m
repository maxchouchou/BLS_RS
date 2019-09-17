% main_BPNN.m
tic

%% 基于反向传播神经网络推荐
%% 说明
% 1、功能：使用反向传播神经网络进行推荐
% 2、运行时间：0.3秒(10000行数据)
% 3、运行需要的输入：ml-1m-train.mat
% 4、运行输出：ml-1m-bpnn
% 5、代码关联：main_Datacreate.m，无后序代码

% BP神经网络模型（无遗传算法优化)
% pmbpnn.m

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

% 提取前10000行
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

%% III.网络创建、训练及仿真测试
%%
fprintf('III.网络创建、训练及仿真测试...\n');
% 1.设置BP神经网络的结构
inputnum = size(P_train,1);               % 输入层神经元个数	
hiddennum = 50;                           % 隐含层神经元个数
outputnum = size(T_train,1);              % 输出层神经元个数
datanum = size(P_train,2);                % 训练样本个数

%%
% 2. 创建网络
% net = newff(p_train,t_train,hiddennum,{'logsig','purelin'},'trainlm','learngdm','crossentropy');
net = newff(p_train,t_train,hiddennum);

%%
% 3. 设置训练参数
net.trainParam.epochs = 50;  % 最大收敛次数 (调优后的)
net.trainParam.goal = 1e-3;   % 训练的最小误差
net.trainParam.show = 10 ;    % 显示间隔
net.trainParam.lr = 0.1;     % 学习率 (调优后的)
net.trainParam.mc=0.1;        % 附加动量因子
%%
% 4. 训练网络
tic
net=train(net,p_train,t_train);
Training_time = toc;
%%
% 5. 仿真测试
tic;
t_sim = sim(net,p_test);
Testing_time = toc;
%%
% 6. 数据反归一化
T_sim=mapminmax('reverse',t_sim,ps_output);
fprintf('\n');

%% IV.性能评价
%%
fprintf('IV.性能评价...\n');

% 1. 计算RMSE
RMSE=sqrt(sum((T_sim-T_test).^2)/testnum);

% 2. 计算决定系数R^2
[~,~,~,~,stats] = regress(T_sim', T_test');
R2 = stats(1);

% 3. Sparsity
sparsity=sum(sum(T_sim==0)/testnum);

% 4. accuracy
meanpre=bsxfun(@minus, T_sim,P_test(5,:));
meantrue=bsxfun(@minus, T_test,P_test(5,:));
accuracy=sum(sum(sign(meanpre)==sign(meantrue))/testnum);

fprintf('\n');

%%
% 2.结果输出
save('ml-1m-bpnn.mat');
fprintf('\n');
toc