function [model]=rs_evaluate(model)
    %% 导入模型数据
    matrix=model.data.matrix;
    ratings=model.data.ratings;
    rating_count=model.data.rating_count;
    M=model.data.M;
    N=model.data.N;
    result_user=model.result.result_user;
    result_item=model.result.result_item;
    pre_user=model.result.pre_user;
    pre_item=model.result.pre_item;
    mean_user=model.data.mean_user;    
    mean_item=model.data.mean_item;
    matrix_null=model.data.matrix_null;
    
    %%  比较评分
    % 基于用户比较 [M,N]
    user_compare=zeros(M,N);
    for i=1:M
        for j =1:N
            user_compare(i,j)=result_user(i,j)-matrix(i,j);
        end
    end
    
    % 基于项目比较 [M,N]
    item_compare=zeros(M,N);
    for i=1:M
        for j =1:N
            item_compare(i,j)=result_item(i,j)-matrix(i,j);
        end
    end   

    %% 验证结果
    % 1. RMSE
    user_error=0;
    item_error=0;
    for i=1:M
        for j =1:N
            user_error=user_error+(user_compare(i,j))^2;
            item_error=item_error+(item_compare(i,j))^2;
        end
    end
    user_RMSe = sqrt(user_error/rating_count);
    item_RMSe = sqrt(item_error/rating_count);

    % 2. R^2
    [~,~,~,~,user_stats] = regress(pre_user(:,3),ratings(:,3));
    user_R2 = user_stats(1);
    [~,~,~,~,item_stats] = regress(pre_item(:,3),ratings(:,3));
    item_R2 = item_stats(1);

    % 3. sparsity
    user_sparsity=sum(sum(pre_user(:,3)==0)/sum(ratings(:,3)~=0));
    item_sparsity=sum(sum(pre_item(:,3)==0)/sum(ratings(:,3)~=0));
    
    % 4. accuracy
    % item
    meanpre_item=bsxfun(@minus,result_item,mean_item);
    meantrue_item=bsxfun(@minus, matrix,mean_item);
    item_accuracy=(sum(sum(sign(meanpre_item)==sign(meantrue_item)))-matrix_null)/rating_count;
    
    % user
    meanpre_user=bsxfun(@minus, result_user,mean_user);
    meantrue_user=bsxfun(@minus, matrix,mean_user);
    user_accuracy=(sum(sum(sign(meanpre_user)==sign(meantrue_user)))-matrix_null)/rating_count;
    
    %% 将结果输入到模型中
    % user
    model.evaluate.user_RMSe = user_RMSe;
    model.evaluate.user_R2 = user_R2;
    model.evaluate.user_sparsity = user_sparsity;
    model.evaluate.user_accuracy=user_accuracy;
    % movie
    model.evaluate.item_RMSe = item_RMSe;
    model.evaluate.item_R2 = item_R2;
    model.evaluate.item_sparsity = item_sparsity;
    model.evaluate.item_accuracy=item_accuracy;
end
