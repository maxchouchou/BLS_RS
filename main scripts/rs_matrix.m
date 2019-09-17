function [model]=rs_matrix(ratings,model)
    minuser=min(ratings(:,1));
    maxuser=max(ratings(:,1));
    minitem=min(ratings(:,2));
    maxitem=max(ratings(:,2));
    itemID=model.data.itemID;
    userID=model.data.userID;
    rating_count=model.data.rating_count;
    print1=['��С�û�Index: ',num2str(minuser),', ����û�Index: ',num2str(maxuser),'\n'];
    print2=['��С��ĿIndex: ',num2str(minitem),', �����ĿIndex: ',num2str(maxitem),'\n'];
    fprintf(print1);
    fprintf(print2);
    
    % �γɳ�ʼ�û����־���
    for i=1:rating_count
        if mod(i,rating_count/5)==0
            percent=i/rating_count;
            fprintf('���ȣ�%.2f%%.\n', percent*100); 
        end   
        userno=ratings(i,1);
        itemno=ratings(i,2);
        score=ratings(i,3);
        matrix(userno,itemno)=score;
        record_index(userno,itemno)=i;
    end
    
    itemIndex=find(sum(matrix,1)~=0);
    userIndex=find(sum(matrix,2)~=0);
    [~,itemInd]=ismember(itemIndex,itemID);
    [~,userInd]=ismember(userIndex,userID);
    
    matrix(:,all(matrix==0,1))=[];
    [M,N]=size(matrix);
    print3=['���־��� | �û���: ',num2str(M),', ��Ŀ��: ',num2str(N),'\n'];
    fprintf(print3);
    
    mean_item=sum(matrix,1)./sum(matrix~=0,1);    
    mean_user=sum(matrix,2)./sum(matrix~=0,2);    
    matrix_null=sum(sum(matrix==0));
    
    model.data.ratings=ratings;
    model.data.matrix=matrix;
    model.data.M=M;
    model.data.N=N;
    model.data.itemIndex=itemIndex;
    model.data.userIndex=userIndex;
    model.data.record_index=record_index;
    model.data.itemInd=itemInd;
    model.data.userInd=userInd;
    model.data.matrix_null=matrix_null;
    model.data.mean_item=mean_item;
    model.data.mean_user=mean_user;
    
end