% main_BLS.m
tic

%% ���ڷ��򴫲��Ľ��Ŀ��ѧϰ�Ƽ�
%% ˵��
% 1�����ܣ�ʹ�øĽ���Ŀ��ѧϰ���������Ƽ�����Ϸ��򴫲�����
% 2������ʱ�䣺0.097268��(10000������)
% 3��������Ҫ�����룺ml-1m-train.mat
% 4�����������ml-1m-blsbp.mat
% 5�����������main_Datacreate.m���޺������

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
load ml-1m-train.mat;
fprintf('\n');

% ��ȡǰ10000��
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

% ����ģ��
bls.data.train_x=p_train';
bls.data.train_y=t_train';
bls.data.test_x=p_test';
bls.data.test_y=T_test';

%% III.���紴����ѵ�����������
%%
fprintf('III.���紴����ѵ�����������...\n');

% 1.ģ�ͽṹ/����
fprintf('��������...\n');
bls.parameter.C = 2^-30;% l2���򻯲��� ϵ�����򻯣��Ա��룩�����򻯲���
bls.parameter.s = 0.8; % ��ǿ�ڵ�������߶�
bls.parameter.N1=3; % ÿ�����ڵ������ڵ�
bls.parameter.N2=2; % ������
bls.parameter.N3=15; % ��ǿ�ڵ���
bls.parameter.epoch=50 ; % ��������
rand('state',1);
fprintf('\n')

% 2.ѵ��+����
fprintf('ѵ��...\n');
bls= bls_train_bp(bls); 
fprintf('\n')

% 3.����һ��
bls.test.pre=mapminmax('reverse',bls.test.output,ps_output);
fprintf('\n');

%% IV.��������
%%
fprintf('IV.��������...\n');
pre=bls.test.pre;
true=bls.data.test_y;
test_x=bls.data.test_x;

% 1. ����RMSE
bls.evaluate.RMSE=sqrt(sum((pre-true).^2)/testnum);

% 2. �������ϵ��R^2
[~,~,~,~,stats] = regress(pre, true);
bls.evaluate.R2 = stats(1);

% 3. Sparsity
bls.evaluate.sparsity=sum(sum(pre==0)/testnum);

% 4. accuracy
meanpre=bsxfun(@minus, pre,P_test(5,:)');
meantrue=bsxfun(@minus, true,P_test(5,:)');
bls.evaluate.accuracy=sum(sum(sign(meanpre)==sign(meantrue))/testnum);

fprintf('\n');

clearvars -except bls;

%% V.�洢ģ��
%% 
save('ml-1m-blsbp.mat');
toc