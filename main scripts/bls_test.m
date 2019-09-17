function [bls] = bls_test(bls)
fprintf('\n')
fprintf('���Կ�ʼ��\n');
    %% ����ģ������
    test_x=bls.data.test_x;
    test_y=bls.data.test_y;
    epochs=bls.parameter.epochs;
    N1=bls.parameter.N1;
    N2=bls.parameter.N2;
    
    %% ѵ������
    Ps = bls.train.PS;
    W=bls.train.W;
    Wh=bls.train.Wh;
    l2=bls.train.L2;
    
    for j=1:epochs
        test_x = zscore(test_x')';  % ��������׼����Z���� [24300 �� 2048 ]
        X_1 = [test_x .1 * ones(size(test_x,1),1)];% ������������� �ұ߼�һ��0.1 [24300 �� 2049]
        Z=zeros(size(test_x,1),N2*N1); % ��ǿ������-������ ��-�������ڵ���100��10 [24300 �� 1000 ]
        for i=1:N2 % ���е���������
            web=bls.train.W_feature{j};
            ps1=Ps{j}(i);
            Z1 = X_1 * web{i};
            Z1  =  mapminmax('apply',Z1',ps1)';
            clear W_feature; 
            clear ps1;
            Z(:,N1*(i-1)+1:N1*i)=Z1;
        end
        
        Z_1 = [Z .1 * ones(size(Z,1),1)];
        H = tansig(Z_1 * Wh{j} * l2{j});
        A=[Z H];

        %% ����׼ȷ��
        Y = A * W{j};
        testing_accuracy =  bls_evaluate(Y,test_y);
        testing_time = toc;
        disp('���β�����ɣ�');
        disp(['���β���ʱ��Ϊ�� ', num2str(testing_time), '��' ]);
        disp(['���β���׼ȷ��Ϊ ', num2str(testing_accuracy * 100), '%' ]);
        
       %% ������Բ���
        bls.test.feature{j}=Z;
        bls.test.enhance{j}=H;
        bls.test.input{j}=A;
        bls.test.weight{j}=W;
        bls.test.output{j}=Y;
        bls.test.TestingAccuracy{j}=testing_accuracy;
        bls.test.Testing_time{j}=testing_time;
        
        %% ��������Ա���
        clear Z1;
        clear X_1;
        clear Z_1;
        clear Wh;
        clear H;
        clear A;
        
    end
fprintf('������ɣ�\n');
end