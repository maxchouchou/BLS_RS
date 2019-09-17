function [Accuracy] = bls_evaluate(y,y_true)
    %% 转换格式
        Y = bls_result(y); 
        Y_true = bls_result(y_true);
        
    %% 准确率
    Accuracy = length(find(Y  == Y_true))/size(Y_true,1);
end