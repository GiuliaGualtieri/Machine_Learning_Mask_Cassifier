# Machine_Learning_Mask_Cassifier
Project Artificial Intelligence & Machine Learning : Mask Classifier

This project aims to develop a Machine Learning Classifier able to detect the mask in any image of face.

The Pipeline of the model is very simple:
- normalization:  MinMaxScaler(feature_range=(0,1));
- pca:            PCA(n_components=0.99); 
- model:          SVC(probability=True).

For training the model it has been chosen a train-test-split, with test size of 0.20 of the total number of sample (50K with mask, 50K no mask).

The choice of the model was selected by comparing 5 differents models on 5folds-CV of the training set. The models compared are:  
-GradientBoost: GradientBoostingClassifier() with n_estimators=50;  
-RandomForest:  RandomForestClassifier() with n_estimators=100;  
-NaiveBayes: GaussianNB();  
-Logistic: LogisticRegression();  
-SVC: SVC(probability=True).  
As one can deduce the SVC achieved the best outcome.

To test the model, run* main.py program on your PC.

*If you have Python version 3.10 you should run as follows
>>>exec(open('main.py').read())

my contact:
  *giulia.gualtieri@mail.polimi.it

I'm a Data Analyst, very passionate to the amazing world of Data.
Contact me if you want to share a challenge with me! It would be a great chance for me to discover something new.
