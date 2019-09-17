function [bls] = bls_train_bp(bls)
    % 输入 
    %---train_x,test_x : 训练集和测试集的输入 
    %---train_y,test_y : 训练集和测试集的输出
    %---We: 随机生成的权重
    %---wh: 随机生成的增强节点
    %----s: 增强节点的伸缩参数
    %----C: 系数正则化的正则系数
    %----N1: 每个窗口的特征节点
    %----N2: 特征窗口的总数
    %----N3: 增强节点的总数
    %----epoch: 迭代次数

    fprintf('\n');
    fprintf('训练开始！\n');
    
    %% 导入模型数据
    train_x=bls.data.train_x;
    train_y=bls.data.train_y;
    test_x=bls.data.test_x;
    test_y=bls.data.test_y;
    N1=bls.parameter.N1;
    N2=bls.parameter.N2;
    N3=bls.parameter.N3;
    C=bls.parameter.C;
    s=bls.parameter.s;
    epoch=bls.parameter.epoch;

    %%%%%%%%%%%%%%特征节点的训练%%%%%%%%%%%%%%
    tic
    train_x = zscore(train_x')'; % 标准化为Z分数
    x_1 = [train_x .1 * ones(size(train_x,1),1)];
    z=zeros(size(train_x,1),N2*N1); % 样本数 所有特征数
    for i=1:N2 % 窗口数
        we_window=2*rand(size(train_x,2)+1,N1)-1; % 特征数×每个窗口特征数
        we{i}=we_window;
        A1 = x_1 * we_window;
        A1 = mapminmax(A1);
        clear we_window;
        web{i}=  bls_sparse(A1,x_1,1e-3,50)';
        
        z1 = x_1 * web{i};
        fprintf(1,'特征节点 窗口中 %f: 输出最大 %f 输出最小 %f\n',i,max(z1(:)),min(z1(:)));
        [z1,ps1]  =  mapminmax(z1',0,1);
        z1 = z1';
        ps(i)=ps1;
        z(:,N1*(i-1)+1:N1*i)=z1;
    end

    %%%%%%%%%%%%%增强节点的训练%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z_1 = [z .1 * ones(size(z,1),1)];
    if N1*N2>=N3
         wh=orth(2*rand(N2*N1+1,N3)-1);
    else
        wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
    end
    h = z_1 *  wh;
    
    % 标准化
    l2 = max(max(h));
    l2 = s/l2;
    fprintf(1,'增强节点：输出最大 %f 输出最小 %f\n',l2,min(h(:)));

    h = tansig(h * l2);
    a=[z h];
    
    %%计算输入层到输出层权重
    w = (a'  *  a+eye(size(a',1)) * (C)) \ ( a'  *  train_y);

    %%%%%%%%%%%%%%%对增强节点权重的反向传播%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = a * w;
    TrainingAccuracy = bls_evaluate(y,train_y); 
    disp(['训练准率 : ', num2str(TrainingAccuracy * 100), ' %' ]);
    
    %%测试
    test_x = zscore(test_x')';
    X_1 = [test_x .1 * ones(size(test_x,1),1)];
    Z=zeros(size(test_x,1),N2*N1);
    for i=1:N2
        web_test=web{i};
        ps1=ps(i);
        Z1 = X_1 * web_test;
        Z1  =  mapminmax('apply',Z1',ps1)';
        clear web_test; 
        clear ps1;
        Z(:,N1*(i-1)+1:N1*i)=Z1;
    end
    
    Z_1 = [Z .1 * ones(size(Z,1),1)]; 
    H = tansig(Z_1 * wh * l2);
    A=[Z H];

    Y = A * w;
    testing_accuracy =  bls_evaluate(Y,test_y);
    disp(['本次测试准确率为 ', num2str(testing_accuracy * 100), '%' ]);

    m=1;
    N=size(z,1);
    wh1=ones(1,N);
    Accuracy_b=zeros(1,5);
    beta_b=zeros(size(w,1),size(w,2),5);
    T3_b=zeros(size(a,1),size(a,2),5);
    wh_b=zeros(size(wh,1),size(wh,2),5);
    l2_b=zeros(size(l2,1),size(l2,2),5);
    T2_b=zeros(size(h,1),size(h,2),5);

    while m<=epoch
        fprintf(['epoch:',num2str(m),'\n']);
        y = a * w;
        W2=w(N1*N2+1:end,:);
        Wh=wh(1:N1*N2,:);
        Wh_beta=wh(N1*N2+1,:);
        T2temp=((1-h.^2));
        tempWh=z'*(((y-train_y)*W2').*T2temp);
        Wh=Wh+0.1*tempWh;
        tempWhbeta=wh1*(((y-train_y)*W2').*T2temp);
        Wh_beta=Wh_beta+0.1*tempWhbeta;
        wh_temp=[Wh; Wh_beta];
        T2_temp = z_1 * wh_temp;
        l2_temp = max(max(T2_temp));
        l2_temp = s/l2_temp;
        T2_temp = tansig(T2_temp * l2_temp);
        T3_temp=[z T2_temp];
        beta_temp = (T3_temp'  *  T3_temp+eye(size(T3_temp',1)) * (C)) \ ( T3_temp'  *  train_y);

        %% 以下代码块用于以下情况:当下一个4次迭代的训练精度不高于第m次迭代的结果
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        xx_temp = T3_temp * beta_temp;
        TrainingAccuracy =  bls_evaluate(xx_temp,train_y);
        disp(['本次训练的准确率为：', num2str(TrainingAccuracy * 100), '%' ]);
        Accuracy=TrainingAccuracy;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if m<=5      
            w=beta_temp; beta_b(:,:,m)=beta_temp;
            a=T3_temp;T3_b(:,:,m)=T3_temp;
            wh=wh_temp;wh_b(:,:,m)=wh_temp;
            l2=l2_temp;l2_b(:,:,m)=l2_temp;
            h=T2_temp;T2_b(:,:,m)=T2_temp;
            Accuracy_b(m)=Accuracy;
            m=m+1;
        else
            beta_b(:,:,1:4)=beta_b(:,:,2:5); beta_b(:,:,5)=beta_temp;
            T3_b(:,:,1:4)=T3_b(:,:,2:5);T3_b(:,:,5)=T3_temp;
            wh_b(:,:,1:4)=wh_b(:,:,2:5);wh_b(:,:,5)=wh_temp;
            l2_b(:,:,1:4)=l2_b(:,:,2:5);l2_b(:,:,5)=l2_temp;
            T2_b(:,:,1:4)=T2_b(:,:,2:5);T2_b(:,:,5)=T2_temp;
            Accuracy_b(1:4)=Accuracy_b(2:5);Accuracy_b(5)=Accuracy;
             [a,index]=max(Accuracy_b);
            if index~=1
                w=beta_temp;
                a=T3_temp;
                wh=wh_temp;
                l2=l2_temp;
                h=T2_temp;
                m=m+1;
            else
                w=beta_b(:,:,1);
                a=T3_b(:,:,1);
                wh=wh_b(:,:,1);
                l2=l2_b(:,:,1);
                h=T2_b(:,:,1);
                break
            end 
        end
        
    end
    
    Training_time = toc;
    disp('训练结束！\n');
    disp(['训练总时间为： ', num2str(Training_time), '秒' ]);

    %%%%%%%%%%%%%%%%%训练精度%%%%%%%%%%%%%%%%%%%%%%%%%%
    y = a * w;
    TrainingAccuracy = bls_evaluate(y , train_y);
    disp(['训练准确率为： ', num2str(TrainingAccuracy * 100), ' %' ]);
    
    bls.train.output=y;
    bls.train.TrainingAccuracy=TrainingAccuracy;
    bls.train.Training_time = Training_time ;
    
    %% TEST
    tic;
    %%%%%%%%%%%%%%%%%%%%%%测试过程%%%%%%%%%%%%%%%%%%%
    % test_x = zscore(test_x')';
    % HH1 = [test_x .1 * ones(size(test_x,1),1)];
    % %clear test_x;
    % yy1=zeros(size(test_x,1),N2*N1);
    % for i=1:N2
    %     beta1=beta11{i};ps1=ps(i);
    %     TT1 = HH1 * beta1;
    %     TT1  =  mapminmax('apply',TT1',ps1)';
    % 
    %     clear beta1; clear ps1;
    %     %yy1=[yy1 TT1];
    %     yy1(:,N1*(i-1)+1:N1*i)=TT1;
    % end
    % clear TT1;clear HH1;
    % HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 
    H = tansig(Z_1 * wh * l2);
    A=[Z H];
    
    %%%%%%%%%%%%%%%%% 测试准确率%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y = A * w;
    TestingAccuracy = bls_evaluate(Y,test_y);

    Testing_time = toc;
    
    disp('本次测试完成！');
    disp(['本次测试时间为： ', num2str( Testing_time), '秒' ]);
    disp(['本次测试准确率为 ', num2str(TestingAccuracy * 100), '%' ]);

    %% 储存测试参数
    bls.test.feature=Z;
    bls.test.enhance=H;
    bls.test.input=A;
    bls.test.weight=w;
    bls.test.output=Y;
    bls.test.Testing_time=Testing_time;
    bls.test.TestingAccuracy=TestingAccuracy;
end

