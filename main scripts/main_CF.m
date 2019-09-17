% main_CF.m
tic

%% ����Эͬ���˵��Ƽ�ϵͳ
%% ˵��
% 1�����ܣ����ں��������Ƽ����룬����Эͬ���˽����Ƽ������еı����Ͳ������洢��model�ṹ�����С�
% 2������ʱ�䣺�˴���ȫ������ʱ��ϳ�����Ҫ1~2Сʱ��
% 3��������Ҫ�����룺data_movies.xlsx��data_users.xlsx��data_ratings.csv��ml-1m-similarity.mat*ע1
% 4�����������ml-1m-model.mat
% 5�������������ǰ����룬�������Ϊmain_Datacreate.m

% ע1��ml-1m-similarity.mat�����ƶȼ����ļ������û����ǰ����III.�������ƶȡ�ȡ�����ƶȼ�������ע��

%% I.��ʼ������
%%
clear;
clc;
warning off;
fprintf('I.��ʼ������...\n');
fprintf('\n');

%% II.���ݵ���
%%
fprintf('II.���ݵ���...\n');

% ����CSV�ļ�
item_data = xlsread('data_movies.xlsx');
users_data = xlsread('data_users.xlsx');
ratings_data = importdata('data_ratings.csv');

% ���ļ�ת��Ϊ����
% item
fprintf('������Ŀ����...\n');
items = item_data(:,2:end); % movies:3883
[m_movie,n_movie]=size(items);
itemID = item_data(:,1)';
model.data.items = items;
model.data.itemID = itemID;
fprintf(['��С��Ŀ���: ',num2str(min(itemID)),', �����Ŀ���: ',num2str(max(itemID)),'\n']);
fprintf(['��Ŀ���� | ��Ŀ��: ',num2str(m_movie),'��������: ',num2str(n_movie),'\n']);
fprintf('\n');

% users
fprintf('�����û�����...\n');
users = users_data(:,2:end); % users:6,040
[m_user,n_user]=size(users);
userID = users_data(:,1);
model.data.users = users;
model.data.userID = userID;
fprintf(['��С�û����: ',num2str(min(userID)),',����û����: ',num2str(max(userID)),'\n']);
fprintf(['�û����� | �û���: ',num2str(m_user),', ������: ',num2str(n_user),'\n']);
fprintf('\n');

% ratings
fprintf('������������...\n');

ratings = ratings_data.data(:,1:end-1); % ratings:1,000,209
rating_count = length(ratings(:,1));
model.data.rating_count = rating_count;

% ת��Ϊmatrix����
fprintf('ת��Ϊmatrix����...\n');
model = rs_matrix(ratings,model);
fprintf('\n');

% ����޹ر���
clearvars -except model;

%% III.�������ƶ�
%% 
% �����ֽ�ע���˼������ƶȵĲ��裬��ֱ��load��ml-1m-similarity.mat���ļ��������߳��ļ�����̡�
% ���ϣ���������ƶ�ֱ��ȡ��ע�ͼ��ɣ��ɲ鿴��Ӧ������
% ���ȫ�������Լ��Ҫ2~3Сʱ������ʱ��

fprintf('III.�������ƶ�...\n');
load('ml-1m-similarity.mat');

%% ��ע�ʹ���
% % �������ƶ�
% fprintf('�����������ƶ�...\n');
% [user_rating] = rs_similarity(model.data.matrix,1); % 1��ʾ����Ϊһ��������㣬���û����û�֮��
% [item_rating] = rs_similarity(model.data.matrix,2); % 2��ʾ����Ϊһ��������㣬����Ŀ����Ŀ֮��
% fprintf('\n');
% 
% % ��Ŀ���ƶ�
% fprintf('������Ŀ�������ƶ�...\n');
% [item_feature] = rs_similarity(model.data.items,1);
% fprintf('\n');
% 
% % �û��������ƶ�
% fprintf('�����û��������ƶ�...\n');
% [user_feature] = rs_similarity(model.data.users,1);
% fprintf('\n');
% 
% % ������ƶȾ���
% fprintf('�����������ƶȾ���...\n');
% fprintf('\n');
% 
% % �û�����
% user_feature = user_feature(model.data.userInd,model.data.userInd);
% user_similarity =( user_rating + user_feature)/2; % �ɿ��ǻ�����ƶȣ����û����ֺ��û��������ߺϲ�
% user_similarity = user_feature; % ����Ŀʹ�õ����û�������Ϊ��������
% 
% % ��Ŀ����
% item_feature = item_feature(model.data.itemInd,model.data.itemInd);
% movie_similarity = (movie_rating + movies_genre)/2; % �ɿ��ǻ�����ƶȣ����û����ֺ��û��������ߺϲ�
% item_similarity = item_feature; % ����Ŀʹ�õ�����Ŀ������Ϊ��������
%% ��ע�ʹ���

% ����ģ��
model.similarity.user_rating = user_rating;
model.similarity.item_rating = item_rating;
model.similarity.item_feature= item_feature;
model.similarity.user_feature = user_feature;
model.similarity.user_similarity = user_similarity;
model.similarity.item_similarity = item_similarity;
clearvars -except model;

%% IV.Ѱ���ھ�
%%
fprintf('IV.Ѱ���ھ�...\n');
nei_rate=0.1; % �ھ�ռ��ϵͳ����ı���
% nei_num=20; % �ھ�������
N_user = round(model.data.M*nei_rate);% ѡ��һ���������ھ�
N_item = round(model.data.N*nei_rate);% ѡ��һ���������ھ�
[model] = rs_neighbor(model,N_user,N_item);% �ҵ������ھӲ������ǵ��������ݴ洢����

clearvars -except model;
fprintf('\n');

%% V.Эͬ�����Ƽ�
fprintf('V.Эͬ�����Ƽ�...\n');
[model] = rs_recommend(model);
fprintf('\n');

%% VI.��֤�Ƽ����
fprintf('VI.��֤�Ƽ����...\n');
[model] = rs_evaluate(model);
fprintf('\n');

%% VII.���ģ��
fprintf('VII.���ģ��...\n');
save('ml-1m-model.mat','model');
fprintf('\n');

toc