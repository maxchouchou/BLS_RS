function [model]=rs_recommend(model)
    %% 导入数据
    % 模型数据
    ratings=model.data.ratings;
    matrix=model.data.matrix;
    user_index=model.data.userIndex;
    item_index=model.data.itemIndex;
    
    % 相似度和邻居
    user_similarity=model.similarity.user_similarity;
    item_similarity=model.similarity.item_similarity;
    user_Neighbor=model.neighbor.user_Neighbor;
    item_Neighbor=model.neighbor.item_Neighbor;
    N_user=model.neighbor.N_user;
    N_item=model.neighbor.N_item;

    %% 推荐
    % 基于用户的协同过滤
    fprintf('基于用户推荐：\n');
    [M,N]=size(matrix);
    for i=1:M
        if mod(i,M/5)==0
            percent=i/M;
            fprintf('进度：%.2f%%.\n', percent*100);
        end
        for j =1:N
            if matrix(i,j)~=0 % 用户i对项目j有评分
                count_rating=0;
                count_number=0;
                for k=1:N_user
                    n_num=user_Neighbor(i,k); % 用户i的邻居k的编号
                    n_rating=matrix(n_num,j); % 邻居k对项目j的评分
                    if n_rating~=0
                        count_rating=n_rating*user_similarity(i,n_num)+count_rating;
                        count_number=count_number+1;
                    end
                end
                result_user(i,j)=count_rating/max(count_number,1);
            end
        end
    end

    % 基于项目的协同过滤
    fprintf('基于项目推荐：\n');
    for i=1:M
        if mod(i,M/5)==0
            percent=i/M;
            fprintf('进度：%.2f%%.\n', percent*100);
        end
        for j =1:N
            if matrix(i,j)~=0 % 用户i对电影j有评分
                count_rating=0;
                count_number=0;
                for k=1:N_item
                    n_num=item_Neighbor(j,k); % 项目j的邻居k的编号
                    n_rating=matrix(i,n_num); % 用户i对邻居k的评分
                    if n_rating~=0
                        count_rating=n_rating*item_similarity(j,n_num)+count_rating;
                        count_number=count_number+1;
                    end
                end
                result_item(i,j)=count_rating/max(count_number,1);
            end
        end
    end

    % 转化为初始评分列表
    fprintf('转化为初始评分列表：')
    count=0;
    rn=length(ratings(:,1));
    for i=1:M
        if mod(i,M/5)==0
            percent=i/M;
            fprintf('进度：%.2f%%.\n', percent*100);
        end
        for j =1:N
            if matrix(i,j)~=0 && count<(rn+1)
                count=count+1;
                pre_user(count,1)=user_index(i);
                pre_user(count,2)=item_index(j);
                pre_user(count,3)=result_user(i,j);                
                pre_item(count,1)=user_index(i);
                pre_item(count,2)=item_index(j);
                pre_item(count,3)=result_item(i,j);
            end
        end
    end
    
   %% 存入模型
    model.result.result_user=result_user;
    model.result.result_item=result_item;
    model.result.pre_user=pre_user;
    model.result.pre_item=pre_item;

end