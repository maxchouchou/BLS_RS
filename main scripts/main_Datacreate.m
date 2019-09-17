% main_Datacreate.m
tic

%% 基于协同过滤推荐创建输入
%% 说明
% 1、功能：构造推荐输入。
% 2、运行时间：70秒左右。
% 3、运行需要的输入：ml-1m-similarity.mat
% 4、运行输出：ml-1m-train.mat
% 5、代码关联：前序代码为main_CF,后续代码为main_BLS、main_BLSBP和main_BPNN

%% I.初始化环境
%%
clear;
clc;
fprintf('I.初始化环境...\n');
warning off

%% II. 数据导入
%%
fprintf('II. 数据导入...\n');
load ml-1m-model.mat;

%%
% 1.数据导入 
% data
data=zeros(model.data.rating_count,8);
ratings=model.data.ratings;
rating_count=model.data.rating_count;
matrix=model.data.matrix;

% user
user_index=model.data.userIndex;
pre_user=model.result.pre_user;
mean_user=model.data.mean_user;
activeness_user=sum(matrix~=0,2);
users=model.data.users;

% item
item_index=model.data.itemIndex;
pre_item=model.result.pre_item;
mean_item=model.data.mean_item;
popularity_item=sum(matrix~=0,1);
items=model.data.items;

fprintf('构建输入变量和输出变量...\n');
for i=1:rating_count
    if mod(i,rating_count/10)==0
        percent=i/rating_count;
        fprintf('进度：%.2f%%.\n', percent*100);
    end
    userindex=ratings(i,1);
    usernum=find(user_index==userindex);
    itemindex=ratings(i,2);
    itemnum=find(item_index==itemindex);
    
    % user
    % 特征属性
    data(i,1)=users(usernum,1);
    data(i,2)=users(usernum,2);
    data(i,3)=users(usernum,3);
    % 行为
    data(i,4)=activeness_user(usernum);
    data(i,5)=mean_user(usernum);
    % 预测
    data(i,6)=pre_user(i);
    
    % item
    % 特征属性
    data(i,7:25)=items(itemnum,1:19);
    % 行为
    data(i,25)=popularity_item(itemnum);
    data(i,26)=mean_item(itemnum);
    % 预测
    data(i,27)=pre_item(i);
    % rating
    data(i,28)=ratings(i,3);
end

input = data(:,1:27);
output= data(:,28);
clearvars -except input output;
save('ml-1m-data.mat','input','output');

%% III.样本划分
%%
fprintf('III. 样本划分...\n');
temp = randperm(size(input,1));
trainratio= 0.6; % 训练集比例
train_num = round(trainratio * size(input,1)); % 训练集样本数
N=size(input,1);            %数据总个数
M=train_num;            %训练数据
% 训练集—样本集的60%
P_train = input(temp(1:train_num),:)';
T_train = output(temp(1:train_num),1)';
% 测试集—样本集的40%
P_test = input(temp((train_num+1):end),:)';
T_test = output(temp((train_num+1):end),1)';

%% IV.数据归一化
%%
fprintf('IV.数据归一化...\n');
[p_train, ps_input] = mapminmax(P_train,0,1);
[t_train, ps_output] = mapminmax(T_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

clear input output;
save('ml-1m-train.mat');
toc