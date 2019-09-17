% main_Datacreate.m
tic

%% ����Эͬ�����Ƽ���������
%% ˵��
% 1�����ܣ������Ƽ����롣
% 2������ʱ�䣺70�����ҡ�
% 3��������Ҫ�����룺ml-1m-similarity.mat
% 4�����������ml-1m-train.mat
% 5�����������ǰ�����Ϊmain_CF,��������Ϊmain_BLS��main_BLSBP��main_BPNN

%% I.��ʼ������
%%
clear;
clc;
fprintf('I.��ʼ������...\n');
warning off

%% II. ���ݵ���
%%
fprintf('II. ���ݵ���...\n');
load ml-1m-model.mat;

%%
% 1.���ݵ��� 
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

fprintf('��������������������...\n');
for i=1:rating_count
    if mod(i,rating_count/10)==0
        percent=i/rating_count;
        fprintf('���ȣ�%.2f%%.\n', percent*100);
    end
    userindex=ratings(i,1);
    usernum=find(user_index==userindex);
    itemindex=ratings(i,2);
    itemnum=find(item_index==itemindex);
    
    % user
    % ��������
    data(i,1)=users(usernum,1);
    data(i,2)=users(usernum,2);
    data(i,3)=users(usernum,3);
    % ��Ϊ
    data(i,4)=activeness_user(usernum);
    data(i,5)=mean_user(usernum);
    % Ԥ��
    data(i,6)=pre_user(i);
    
    % item
    % ��������
    data(i,7:25)=items(itemnum,1:19);
    % ��Ϊ
    data(i,25)=popularity_item(itemnum);
    data(i,26)=mean_item(itemnum);
    % Ԥ��
    data(i,27)=pre_item(i);
    % rating
    data(i,28)=ratings(i,3);
end

input = data(:,1:27);
output= data(:,28);
clearvars -except input output;
save('ml-1m-data.mat','input','output');

%% III.��������
%%
fprintf('III. ��������...\n');
temp = randperm(size(input,1));
trainratio= 0.6; % ѵ��������
train_num = round(trainratio * size(input,1)); % ѵ����������
N=size(input,1);            %�����ܸ���
M=train_num;            %ѵ������
% ѵ��������������60%
P_train = input(temp(1:train_num),:)';
T_train = output(temp(1:train_num),1)';
% ���Լ�����������40%
P_test = input(temp((train_num+1):end),:)';
T_test = output(temp((train_num+1):end),1)';

%% IV.���ݹ�һ��
%%
fprintf('IV.���ݹ�һ��...\n');
[p_train, ps_input] = mapminmax(P_train,0,1);
[t_train, ps_output] = mapminmax(T_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

clear input output;
save('ml-1m-train.mat');
toc