function [bls] = bls_train(bls)
    % ������
    %----s: ��ǿ�ڵ����������
    %----C: ϵ�����򻯣��Ա��룩�����򻯲���
    %----N1: ÿ�����ڵĽڵ���
    %----N2: ������
    %----N3: ��ǿ�ڵ���
    
    % ѵ������
    %---we: �����ڵ�������ɵ�Ȩ��
    %---wh: ��ǿ�ڵ�������ɵ�Ȩ��    
    fprintf('\n');
    fprintf('ѵ����ʼ��\n');
    
    %% ����ģ������
    train_x=bls.data.train_x;
    train_y=bls.data.train_y;
    epochs=bls.parameter.epochs;
    N1=bls.parameter.N1;
    N2=bls.parameter.N2;
    N3=bls.parameter.N3;
    C=bls.parameter.C;
    s=bls.parameter.s;
    
    for j=1:epochs
        tic % ��ʱ��ʼ
        %% �����ڵ�                       
        train_x = zscore(train_x')'; % ��������׼����Z���� [24300 �� 2048 ]
        x_1 =[train_x 0.1 * ones(size(train_x,1),1)]; % ������������� H1 = train_x �ұ߼�һ��0.1 [24300 �� 2049]
        z=zeros(size(train_x,1),N2*N1); % �����ڵ��ʼ��-Ϊ0����-������ ��-�������ڵ���100��10 [24300 �� 1000 ]

        for i=1:N2 % ����ÿ������
            we_window=2*rand(size(train_x,2)+1,N1)-1;% �����ڵ�Ȩ�� [2049 �� 100] 
            we{i}=we_window; % ÿ�����������ڵ��Ȩ�غ���ֵ
            A1 = x_1 * we_window; % ӳ�����������Ȩֵ = �������� * Ȩ��  [24300 �� 100] 
            A1 = mapminmax(A1); % ��׼��
            clear we_window; % ���Ȩ��
            web{i}  =  bls_sparse(A1,x_1,1e-3,50)'; % ϡ���Ա���
            
            z1 = x_1 * web{i}; % ӳ�������������ֵ = �������� * ��ֵ  [24300 �� 100] 
            fprintf(1,'�����ڵ㴰�� %f: ������Ϊ�� %f ��СΪ�� %f\n',i,max(z1(:)),min(z1(:)));
            [z1,ps1]  =  mapminmax(z1',0,1); % ��׼����0-1
            z1 = z1'; % ת��
            ps(i)=ps1; % ��¼��׼�������
            z(:,N1*(i-1)+1:N1*i)=z1;% �Ե�һ�����ڵ������ڵ���и�ֵ
        end

        %% ��ǿ�ڵ�
        z_1 = [z .1 * ones(size(z,1),1)];  % ��������ǿ�ڵ���� H2 = y �ұ߼�һ��0.1 [24300 �� 2049]
        if N1*N2>=N3 % �����ڵ��������ڵ�����ǿ�ڵ�����
            wh=orth(2*rand(N2*N1+1,N3)-1); % orth �����淶������-��ǿȨֵ��ֵ orth([1001 �� 1000])
        else
            wh=orth(2*rand(N2*N1+1,N3)'-1)'; % ���-��ǿȨֵ��ֵ orth([1001 �� 1000])
        end
        h = z_1 * wh; % ���������* Ȩֵ��ֵ [24300 �� 1000]

        % ��׼��
        l2 = max(max(h)); % �ҵ����ֵ
        l2 = s/l2; % ��������/���ֵ
        fprintf(1,'��ǿ�ڵ㣺������ֵΪ %f ��СֵΪ %f\n',l2,min(h(:)));

        h = tansig(h * l2); % ����� tansig  [24300 �� 1000 ]
        a=[z h]; % �����������������T3 [24300 �� 2000 ]

       %% ��������㵽�����Ȩ��
        w = (a' *  a+eye(size(a',1)) * (C)) \ ( a'  *  train_y); % ��ֵ ���뵽�������ֵ [2000 �� 5]
        
       %% ����ѵ��ʱ��
        training_time = toc;
        
        disp('����ѵ����ɣ�');
        disp(['����ѵ��ʱ��Ϊ�� ', num2str(training_time), ' ��' ]);

        %% ����ѵ��׼ȷ��
        y = a * w; % [24300 �� 5] ÿ��������Ӧ�����
        % �Ƚ�
        training_accuracy = bls_evaluate(y,train_y);      
        disp(['����ѵ��׼ȷ��Ϊ�� ', num2str(training_accuracy * 100), ' %' ]);
        
        %% ����ѵ������
        bls.train.We{j}=we;
        bls.train.W_feature{j}=web;
        bls.train.PS{j}=ps;
        bls.train.Z{j}=z;
        bls.train.L2{j}=l2;
        bls.train.Wh{j}=wh;
        bls.train.H{j}=h;
        bls.train.A{j}=a;
        bls.train.W{j}=w;
        bls.train.Y{j}=y;
        bls.train.TrainingAccuracy{j}=training_accuracy;
        bls.train.TrainingTime{j}=training_time;
        
        %% ������̱���
        clear x_1;
        clear Z1;
        clear z_1;
        clear h;
        clear a;
        
    end % ��������
    fprintf('ѵ����ɣ�\n');
end