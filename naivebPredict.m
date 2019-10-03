function [p]=naivebPredict(phik,phi,X)
%   naivebPredict returns a vector of predictions using a trained naive bayes probabilities 
%   phik and phi. X is a mxn matrix where there each example is a row.   
%   predictions pred is a m x 1 column of predictions of {0, 1} values.


% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end
m = size(X, 1); 
predictX=zeros(m,1);

%retrieve the probabilities vectors
phik0=phik(1,:);
phik1=phik(2,:);
phiy1=phi(2);
phiy0=phi(1);
% compare log ?p(x|y=0)+log p(y=0)  and  log ?p(x|y=1)+log p(y=1)  
predictX=X*log(phik1')+log(phiy1)>X*log(phik0')+log(phiy0);
p=predictX;
