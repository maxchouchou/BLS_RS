% main_BPNN.m
tic

%% ���ڷ��򴫲��������Ƽ�
%% ˵��
% 1�����ܣ�ʹ�÷��򴫲�����������Ƽ�
% 2������ʱ�䣺0.3��(10000������)
% 3��������Ҫ�����룺ml-1m-train.mat
% 4�����������ml-1m-bpnn
% 5�����������main_Datacreate.m���޺������

% BP������ģ�ͣ����Ŵ��㷨�Ż�)
% pmbpnn.m

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

%% III.���紴����ѵ�����������
%%
fprintf('III.���紴����ѵ�����������...\n');
% 1.����BP������Ľṹ
inputnum = size(P_train,1);               % �������Ԫ����	
hiddennum = 50;                           % ��������Ԫ����
outputnum = size(T_train,1);              % �������Ԫ����
datanum = size(P_train,2);                % ѵ����������

%%
% 2. ��������
% net = newff(p_train,t_train,hiddennum,{'logsig','purelin'},'trainlm','learngdm','crossentropy');
net = newff(p_train,t_train,hiddennum);

%%
% 3. ����ѵ������
net.trainParam.epochs = 50;  % ����������� (���ź��)
net.trainParam.goal = 1e-3;   % ѵ������С���
net.trainParam.show = 10 ;    % ��ʾ���
net.trainParam.lr = 0.1;     % ѧϰ�� (���ź��)
net.trainParam.mc=0.1;        % ���Ӷ�������
%%
% 4. ѵ������
tic
net=train(net,p_train,t_train);
Training_time = toc;
%%
% 5. �������
tic;
t_sim = sim(net,p_test);
Testing_time = toc;
%%
% 6. ���ݷ���һ��
T_sim=mapminmax('reverse',t_sim,ps_output);
fprintf('\n');

%% IV.��������
%%
fprintf('IV.��������...\n');

% 1. ����RMSE
RMSE=sqrt(sum((T_sim-T_test).^2)/testnum);

% 2. �������ϵ��R^2
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
% 2.������
save('ml-1m-bpnn.mat');
fprintf('\n');
toc