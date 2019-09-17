function [model]=rs_neighbor(model,N_user,N_item)
    user_similarity=model.similarity.user_similarity;
    item_similarity=model.similarity.item_similarity;
    % user
    [~,N_user_index]=sort(user_similarity,1,'descend');
    user_Neighbor=N_user_index(2:N_user+1,:)';

    % movie
    [~,N_item_index]=sort(item_similarity,1,'descend');
    item_Neighbor=N_item_index(2:N_item+1,:)';
    
    model.neighbor.user_Neighbor = user_Neighbor;
    model.neighbor.item_Neighbor = item_Neighbor;
    model.neighbor.N_user = N_user;
    model.neighbor.N_item = N_item;
end