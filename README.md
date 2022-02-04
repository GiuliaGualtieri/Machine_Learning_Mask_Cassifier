# Machine_Learning_Mask_Cassifier
Project Artificial Intelligence & Machine Learning : Mask Classifier

This project aims to develop a Machine Learning Classifier able to detect the mask in any image of visage.

The Pipeline of the model is very simple:
- normalization:  MinMaxScaler(feature_range=(0,1));
- pca:            PCA(0.99); 
- model:          SVC.

For training the model it has been chosen a train-test-split, with test size of 0.20 of the total numer of sample (%0K with mask, 50K no mask).


To test the model, run* main.py program on your PC.

*If you have Python version 3.10 you should run as follows
>>>exec(open('main.py').read())

my contact:
  *giulia.gualtieri@mail.polimi.it

I'm a Data Analyst, very passionate to data. 
Contact me if you want to share a challenge with me! It would be very glad to me having the possibility to find out something new about the amazing world of Data. 
