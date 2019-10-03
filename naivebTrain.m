function [phik,phi] = naivebTrain(X, Y)
%   this function trains a Naive Bayes classifier and returns calculated probabilities : p(k|y=0)
%   p(k|y=1),p(y=0) and p(y=1).  X is the matrix of training examples.
%   Each row is a training example, and the jth column holds the 
%   jth feature.  Y is a column matrix containing 1 for positive examples 
%   and 0 for negative examples.   
                     
% Data parameters

m = size(X, 1);
n = size(X, 2);
%m number of samples and n number of features

%phi_ky0k =probability_(k/y=0) and phi_ky1=probability_(k/y=1) where k is the k_th feature of the 1899 features from the vocabList
%phi_y0=probability(y=0)  and phi_y1=probability(y=1)
phi_ky0=zeros(1,n);
phi_ky1=zeros(1,n);
% Z0 and Z1 are auxilary matrixes
%Z1 matrix of 0 and 1s ,1 if X(i,j)=1 and y(i)=1 and 0 if not. It detects spam features.i is the i_th sample
%Z0 matrix of 0 and 1s,1 if X(i,j)=1 and y(i)=0 and 0 if not.It detects non spam features.i is the i_th sample
Z1=Y*ones(1,n).*X;
Z0=(1-Y)*ones(1,n).*X; %m*n dimension
% 1-Y represents the non spam emails whereas Y represents spam email, Y vector of 0s and 1s.
phi_ky0=(sum(Z0)+1)./(sum(X')*(1-Y)+n);
phi_ky1=(sum(Z1)+1)./(sum(X')*Y+n); 
phi_y1=sum(Y)/m;
phi_y0=sum(1-Y)/m;


%phik=[phi_ky0,phi_ky1]
phik=zeros(2,n);
phik(1,:)=phi_ky0;
phik(2,:)=phi_ky1;
phi=[phi_y0,phi_y1];




