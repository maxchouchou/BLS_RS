function [bls] = bls_test(bls)
fprintf('\n')
fprintf('测试开始！\n');
    %% 导入模型数据
    test_x=bls.data.test_x;
    test_y=bls.data.test_y;
    epochs=bls.parameter.epochs;
    N1=bls.parameter.N1;
    N2=bls.parameter.N2;
    
    %% 训练参数
    Ps = bls.train.PS;
    W=bls.train.W;
    Wh=bls.train.Wh;
    l2=bls.train.L2;
    
    for j=1:epochs
        test_x = zscore(test_x')';  % 输入矩阵标准化：Z分数 [24300 × 2048 ]
        X_1 = [test_x .1 * ones(size(test_x,1),1)];% 增广后的输入矩阵 右边加一列0.1 [24300 × 2049]
        Z=zeros(size(test_x,1),N2*N1); % 增强矩阵：行-样本数 列-总特征节点数100×10 [24300 × 1000 ]
        for i=1:N2 % 所有的特征窗口
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

        %% 测试准确率
        Y = A * W{j};
        testing_accuracy =  bls_evaluate(Y,test_y);
        testing_time = toc;
        disp('本次测试完成！');
        disp(['本次测试时间为： ', num2str(testing_time), '秒' ]);
        disp(['本次测试准确率为 ', num2str(testing_accuracy * 100), '%' ]);
        
       %% 储存测试参数
        bls.test.feature{j}=Z;
        bls.test.enhance{j}=H;
        bls.test.input{j}=A;
        bls.test.weight{j}=W;
        bls.test.output{j}=Y;
        bls.test.TestingAccuracy{j}=testing_accuracy;
        bls.test.Testing_time{j}=testing_time;
        
        %% 清除过程性变量
        clear Z1;
        clear X_1;
        clear Z_1;
        clear Wh;
        clear H;
        clear A;
        
    end
fprintf('测试完成！\n');
end