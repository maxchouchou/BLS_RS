function [Accuracy] = bls_evaluate(y,y_true)
    %% ת����ʽ
        Y = bls_result(y); 
        Y_true = bls_result(y_true);
        
    %% ׼ȷ��
    Accuracy = length(find(Y  == Y_true))/size(Y_true,1);
end