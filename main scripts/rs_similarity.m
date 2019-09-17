function [similarity] = rs_similarity(matrix,index)
    % index:1 row-object col-features 2 row-features col-object
    if index==1
        [m,n] = size(matrix);
        similarity = zeros(m);
    elseif index==2
        matrix=matrix';
        [m,n] = size(matrix);
        similarity = zeros(m);
    else
        fprintf(['index = ',num2str(index),' is wrong, please enter 1 or 2!']);
        similarity=[];
        return
    end
    
    % Compute similarity matrix
    fprintf('Compute similarity matrix£º\n')
    for i1 = 1:m
        if mod(i1,m/5)==0
            percent=i1/m;
            fprintf('½ø¶È£º%.2f%%.\n', percent*100); 
        end   
        for i2 = 1:m
            object1 = matrix(i1,:);
            pbject2 = matrix(i2,:);
            count = 0;
            check = 0;
            count_semi = 0;          
            % for each rating
            for j = 1:n
                if object1(j) ~= 0 && pbject2(j) ~= 0
                    count_semi = count_semi+1;
                    if object1(j) == pbject2(j)
                        check = 1;
                        count = count+check;
                    else
                        check = abs(object1(j)-pbject2(j))/max(object1(j),pbject2(j));
                        count = count+check;
                    end
                end
                similarity(i1,i2) = count/max(count_semi,1);
            end
        end
    end
    
end
    