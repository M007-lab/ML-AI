

%% Initialization
%clear ; close all; clc

%% ==================== Part 1: Email Preprocessing ====================

%An exmaple of email preprocessing 
fprintf('\nPreprocessing sample spam (spamSample1.txt)\n');

% Extract Features
file_contents = readFile('spamSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================


fprintf('\nExtracting features from sample spam (spamSample1.txt)\n');

% Extract Features
file_contents = readFile('spamSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;


load('spamTrain.mat');
%%  ===========  SVM  Linear Kernel vs Gaussian kernel       ===========

fprintf('\nComparing Linear SVM vs Gaussian SVM for different training:test ratios \n')
   
C = 0.1;
% ratio training:test of 600:400
model = svmTrain(X(1:600,:), y(1:600), C, @gaussianKernel);
p = svmPredict(model, X(601:1000,:));
fprintf(' Accuracy 600:400 Gaussian Kernel : %f\n', mean(double(p == y(601:1000))) * 100);
model = svmTrain(X(1:600,:), y(1:600), C, @linearKernel);
p = svmPredict(model, X(601:1000,:));
fprintf(' Accuracy 600:400 Linear Kernel : %f\n', mean(double(p == y(601:1000))) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ratio training:test of 700:300
model = svmTrain(X(1:700,:), y(1:700), C, @gaussianKernel);
p = svmPredict(model, X(701:1000,:));
fprintf(' Accuracy 700:300 Gaussian Kernel : %f\n', mean(double(p == y(701:1000))) * 100);
model = svmTrain(X(1:700,:), y(1:700), C, @linearKernel);
p = svmPredict(model, X(701:1000,:));
fprintf(' Accuracy 700:300 Linear Kernel : %f\n', mean(double(p == y(701:1000))) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ratio training:test of 800:200
model = svmTrain(X(1:800,:), y(1:800), C, @gaussianKernel);
p = svmPredict(model, X(801:1000,:));
fprintf(' Accuracy 800:200 Gaussian Kernel : %f\n', mean(double(p == y(801:1000))) * 100);
model = svmTrain(X(1:800,:), y(1:800), C, @linearKernel);
p = svmPredict(model, X(801:1000,:));
fprintf(' Accuracy 800:200 Linear Kernel : %f\n', mean(double(p == y(801:1000))) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% =========== Part 3: Train Linear SVM and NB for Spam Classification ========

%Train our models using a training dataset then measure the accuracy,precision and recall on the training set
fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

% we select the most optimum C parameter=0.1 and we use linearKernel measure 
C = 0.1;
% our training dataset is a matrix X of size 4000 samples each with 1899 features 4000x1899
model = svmTrain(X, y, C, @gaussianKernel);

% we predict the loaded training dataset  X using the the obtained model
p = svmPredict(model, X);

%we get the true predictions (positive and negative) when p  is equal to y 
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%we get the true positive results by calculating scalar product between y and p p'*y
%sum(p) = true positive + false positive
fprintf(' Training Precision: %f\n', (p'*y)/sum(p)*100);

%sum(y) = true positive + false negative
fprintf(' Training Recall: %f\n', (p'*y)/sum(y) * 100);
fprintf('\nTraining Naive Bayes (Spam Classification)\n')

%phik =[phi_(k/y=0),phi_(k/y=1)] where k is the k_th feature of the 1899 features from the vocabList
%phi=[phi(y=0),phi(y=1)]
[phik,phi]= naivebTrain(X,y);

p=naivebPredict(phik,phi,X);
%m is number of samples
m=size(X,1);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf(' Training Precision: %f\n', (p'*y)/sum(p)*100);
fprintf(' Training Recall: %f\n', (p'*y)/sum(y) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 4: Test Spam Classification ================
%Apply our trained model on the testset and get accuracy,precision and recall metrics :cross_validation task
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
fprintf(' Test Precision: %f\n', (p'*ytest)/sum(p)*100);
fprintf(' Test Recall: %f\n', (p'*ytest)/sum(ytest) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nEvaluating the trained naive bayes on a test set ...\n')

p = naivebPredict(phik,phi,Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
fprintf(' Test Precision: %f\n', (p'*ytest)/sum(p)*100);
fprintf(' Test Recall: %f\n', (p'*ytest)/sum(ytest) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Top Predictors of Spam ====================

% In this part,we infer the top spam vocabulary words based on the trained model
% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam SVM: \n');
for i = 1:15
   fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
% Sort the probabilities of spam: the mean of (phi_(k/y=1),1-phi_(k/y=0))
[weight, idx] = sort(((phik(2,:))+1-(phik(1,:)))/2, 'descend');


fprintf('\nTop predictors of spam Naive Bayes: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% =================== Part 6:  Emails =====================
%try the models on an exmaple of spam email4
fprintf('\nTry the models on a spam email example: \n');
filename = 'spamSample2.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);
fprintf('\nProcessed %s\n\nSpam Classification SVM: %d\n', filename, p);

p=naivebPredict(phik,phi,x);
fprintf('\nSpam Classification NB: %d\n', p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

