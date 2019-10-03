This project presents the resulting output of comparing Naive Bayes vesus Support Vector Machines performances on Email classification. It builds both models from scratch without using external libraries.

Files included in this project


main.m - Octave/Matlab script main function 

emailSample1.txt - Example email 1 
emailSample2.txt - Example email 2 
spamSample1.txt - Example spam 1 
spamSample2.txt - Example spam 2 

vocab.txt - vocabulary list doc
readFile.m - function to read files
getVocabList.m - function to read and convert the voca.txt file
porterStemmer.m - stemming function
processEmail.m - email preprocessing function
svmTrain.m - SVM training function
svmPredict.m - SVM prediction function
spamTrain.mat- training dataset
spamTest.mat - test dataset

naivebTrain.m - Naive Bayes training function
naivebPredict.m - Naive Bayes  prediction function
 
linearKernel.m - Linear kernel for SVM 
gaussianKernel.m - Gaussian kernel for SVM 


In order to launch the project you should call the main function in the Octave/Matlab command window.

