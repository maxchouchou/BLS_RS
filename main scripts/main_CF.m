% main_CF.m
tic

%% 基于协同过滤的推荐系统
%% 说明
% 1、功能：用于后续创建推荐输入，基于协同过滤进行推荐。所有的变量和参数均存储在model结构对象中。
% 2、运行时间：此代码全部运行时间较长，需要1~2小时。
% 3、运行需要的输入：data_movies.xlsx、data_users.xlsx、data_ratings.csv、ml-1m-similarity.mat*注1
% 4、运行输出：ml-1m-model.mat
% 5、代码关联：无前序代码，后序代码为main_Datacreate.m

% 注1：ml-1m-similarity.mat：相似度计算文件，如果没有请前往“III.计算相似度”取消相似度计算代码的注释

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

% 导入CSV文件
item_data = xlsread('data_movies.xlsx');
users_data = xlsread('data_users.xlsx');
ratings_data = importdata('data_ratings.csv');

% 将文件转化为矩阵
% item
fprintf('导入项目数据...\n');
items = item_data(:,2:end); % movies:3883
[m_movie,n_movie]=size(items);
itemID = item_data(:,1)';
model.data.items = items;
model.data.itemID = itemID;
fprintf(['最小项目编号: ',num2str(min(itemID)),', 最大项目编号: ',num2str(max(itemID)),'\n']);
fprintf(['项目矩阵 | 项目数: ',num2str(m_movie),'，特征数: ',num2str(n_movie),'\n']);
fprintf('\n');

% users
fprintf('导入用户数据...\n');
users = users_data(:,2:end); % users:6,040
[m_user,n_user]=size(users);
userID = users_data(:,1);
model.data.users = users;
model.data.userID = userID;
fprintf(['最小用户编号: ',num2str(min(userID)),',最大用户编号: ',num2str(max(userID)),'\n']);
fprintf(['用户矩阵 | 用户数: ',num2str(m_user),', 特征数: ',num2str(n_user),'\n']);
fprintf('\n');

% ratings
fprintf('导入评分数据...\n');

ratings = ratings_data.data(:,1:end-1); % ratings:1,000,209
rating_count = length(ratings(:,1));
model.data.rating_count = rating_count;

% 转化为matrix矩阵
fprintf('转化为matrix矩阵...\n');
model = rs_matrix(ratings,model);
fprintf('\n');

% 清除无关变量
clearvars -except model;

%% III.计算相似度
%% 
% 本部分将注释了计算相似度的步骤，请直接load“ml-1m-similarity.mat”文件来避免冗长的计算过程。
% 如果希望计算相似度直接取消注释即可，可查看对应函数。
% 如果全部运算大约需要2~3小时的运算时间

fprintf('III.计算相似度...\n');
load('ml-1m-similarity.mat');

%% 已注释代码
% % 评分相似度
% fprintf('计算评分相似度...\n');
% [user_rating] = rs_similarity(model.data.matrix,1); % 1表示以行为一个对象计算，即用户与用户之间
% [item_rating] = rs_similarity(model.data.matrix,2); % 2表示以列为一个对象计算，即项目与项目之间
% fprintf('\n');
% 
% % 项目相似度
% fprintf('计算项目特征相似度...\n');
% [item_feature] = rs_similarity(model.data.items,1);
% fprintf('\n');
% 
% % 用户特征相似度
% fprintf('计算用户特征相似度...\n');
% [user_feature] = rs_similarity(model.data.users,1);
% fprintf('\n');
% 
% % 结合相似度矩阵
% fprintf('计算总体相似度矩阵...\n');
% fprintf('\n');
% 
% % 用户特征
% user_feature = user_feature(model.data.userInd,model.data.userInd);
% user_similarity =( user_rating + user_feature)/2; % 可考虑混合相似度，将用户评分和用户特征二者合并
% user_similarity = user_feature; % 本项目使用的是用户特征作为计算依据
% 
% % 项目特征
% item_feature = item_feature(model.data.itemInd,model.data.itemInd);
% movie_similarity = (movie_rating + movies_genre)/2; % 可考虑混合相似度，将用户评分和用户特征二者合并
% item_similarity = item_feature; % 本项目使用的是项目特征作为计算依据
%% 已注释代码

% 存入模型
model.similarity.user_rating = user_rating;
model.similarity.item_rating = item_rating;
model.similarity.item_feature= item_feature;
model.similarity.user_feature = user_feature;
model.similarity.user_similarity = user_similarity;
model.similarity.item_similarity = item_similarity;
clearvars -except model;

%% IV.寻找邻居
%%
fprintf('IV.寻找邻居...\n');
nei_rate=0.1; % 邻居占总系统主体的比例
% nei_num=20; % 邻居总数量
N_user = round(model.data.M*nei_rate);% 选出一定比例的邻居
N_item = round(model.data.N*nei_rate);% 选出一定比例的邻居
[model] = rs_neighbor(model,N_user,N_item);% 找到相似邻居并将他们的评分数据存储起来

clearvars -except model;
fprintf('\n');

%% V.协同过滤推荐
fprintf('V.协同过滤推荐...\n');
[model] = rs_recommend(model);
fprintf('\n');

%% VI.验证推荐结果
fprintf('VI.验证推荐结果...\n');
[model] = rs_evaluate(model);
fprintf('\n');

%% VII.输出模型
fprintf('VII.输出模型...\n');
save('ml-1m-model.mat','model');
fprintf('\n');

toc