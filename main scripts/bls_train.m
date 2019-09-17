function [bls] = bls_train(bls)
    % 超参数
    %----s: 增强节点的收缩参数
    %----C: 系数正则化（自编码）的正则化参数
    %----N1: 每个窗口的节点数
    %----N2: 窗口数
    %----N3: 增强节点数
    
    % 训练参数
    %---we: 特征节点随机生成的权重
    %---wh: 增强节点随机生成的权重    
    fprintf('\n');
    fprintf('训练开始！\n');
    
    %% 导入模型数据
    train_x=bls.data.train_x;
    train_y=bls.data.train_y;
    epochs=bls.parameter.epochs;
    N1=bls.parameter.N1;
    N2=bls.parameter.N2;
    N3=bls.parameter.N3;
    C=bls.parameter.C;
    s=bls.parameter.s;
    
    for j=1:epochs
        tic % 计时开始
        %% 特征节点                       
        train_x = zscore(train_x')'; % 输入矩阵标准化：Z分数 [24300 × 2048 ]
        x_1 =[train_x 0.1 * ones(size(train_x,1),1)]; % 增广后的输入矩阵 H1 = train_x 右边加一列0.1 [24300 × 2049]
        z=zeros(size(train_x,1),N2*N1); % 特征节点初始化-为0：行-样本数 列-总特征节点数100×10 [24300 × 1000 ]

        for i=1:N2 % 对于每个窗口
            we_window=2*rand(size(train_x,2)+1,N1)-1;% 特征节点权重 [2049 × 100] 
            we{i}=we_window; % 每个窗口特征节点的权重和阈值
            A1 = x_1 * we_window; % 映射后的输入矩阵权值 = 输入数据 * 权重  [24300 × 100] 
            A1 = mapminmax(A1); % 标准化
            clear we_window; % 清空权重
            web{i}  =  bls_sparse(A1,x_1,1e-3,50)'; % 稀疏自编码
            
            z1 = x_1 * web{i}; % 映射后的输入矩阵阈值 = 输入数据 * 阈值  [24300 × 100] 
            fprintf(1,'特征节点窗口 %f: 输出最大为： %f 最小为： %f\n',i,max(z1(:)),min(z1(:)));
            [z1,ps1]  =  mapminmax(z1',0,1); % 标准化到0-1
            z1 = z1'; % 转置
            ps(i)=ps1; % 记录标准化的情况
            z(:,N1*(i-1)+1:N1*i)=z1;% 对第一个窗口的特征节点进行赋值
        end

        %% 增强节点
        z_1 = [z .1 * ones(size(z,1),1)];  % 增广后的增强节点矩阵 H2 = y 右边加一列0.1 [24300 × 2049]
        if N1*N2>=N3 % 特征节点总数大于等于增强节点总数
            wh=orth(2*rand(N2*N1+1,N3)-1); % orth 正交规范后的输出-增强权值阈值 orth([1001 × 1000])
        else
            wh=orth(2*rand(N2*N1+1,N3)'-1)'; % 输出-增强权值阈值 orth([1001 × 1000])
        end
        h = z_1 * wh; % 隐射后的输出* 权值阈值 [24300 × 1000]

        % 标准化
        l2 = max(max(h)); % 找到最大值
        l2 = s/l2; % 收缩参数/最大值
        fprintf(1,'增强节点：输出最大值为 %f 最小值为 %f\n',l2,min(h(:)));

        h = tansig(h * l2); % 激活函数 tansig  [24300 × 1000 ]
        a=[z h]; % 输入层整体的输入矩阵T3 [24300 × 2000 ]

       %% 计算输入层到输出层权重
        w = (a' *  a+eye(size(a',1)) * (C)) \ ( a'  *  train_y); % 阈值 输入到输出的阈值 [2000 × 5]
        
       %% 计算训练时间
        training_time = toc;
        
        disp('本次训练完成！');
        disp(['本次训练时间为： ', num2str(training_time), ' 秒' ]);

        %% 计算训练准确率
        y = a * w; % [24300 × 5] 每个样本对应的输出
        % 比较
        training_accuracy = bls_evaluate(y,train_y);      
        disp(['本次训练准确度为： ', num2str(training_accuracy * 100), ' %' ]);
        
        %% 储存训练参数
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
        
        %% 清除过程变量
        clear x_1;
        clear Z1;
        clear z_1;
        clear h;
        clear a;
        
    end % 迭代结束
    fprintf('训练完成！\n');
end