# Credit Card Fraud Detection
* This repository holds an attempt to predict whether the credit transaction was normal or a fraud from Kaggle challenge 'Credit Card Fraud Detection.'
* link - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Overview


 * **Definition of the tasks / challenge:**  The task is to predict the transcation done to be normal or a fraud with predictive models.
 
 * **Approach:** The approach in this repository formulates the problem by using XGBClassifier and LogisticRegression from the time-series data available about transactions. The metric considered is f1 score with confusion matrix. We compared the performance of 2 different classifiers.
 
 * **Summary of the performance achieved:** Our best model was able to predict the transcation to be normal or fraud with the metric(f1 score) for Undersample Data: 0.93, Oversample Data: 0.999, Imbalanced(Original) Data: 0.099.

## Summary of Workdone

### Data

* Data:
  * Files:
   * File descriptions:
       * creditcard.csv - It contains numerical information of transactions done in 2 days, which are the result of a PCA transformation.
      
  * Type:
    * Input: CSV file of following features:
      * Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
      * V1-V28: Numerical features. Information is not provided due to privacy reasons.
      * Amount - Amount of transaction.
      * Class - It contains data on whether transaction was normal or a fraud. 1 for fraudulent and 0 otherwise.
      
    * Output: Class - It contains data on whether transaction was normal or a fraud. 1 for fraudulent and 0 otherwise.
    
  * Size: 150.83 MB
  
  * Instances (Train, Test, Validation Split): 
     * Undersample data points: 984. 788 for training, 196 for testing.
     * Oversample data points: 568630. 454904 for training, 113726 for testing.
     * Imbalance(Original) data points: 284807. 227846 for training, 56961 for testing.

### Preprocessing / Clean up

* Removed outliers for V10, V12, V14 features.

* Scaled Time and Amount with RobustScaler because it is less prone to outliers.

* The dataset was imbalanced with 0.2% (492) fraud transactions and 0.98% normal transactions. Thus, following sampling was done.
  * Random UnderSampling: 492 fraud and non-frauds.
  * OverSampling: 284315 non-frauds and fraud.
     
### Data Visualization

Histogram, Box plots, etc were used for data visualization in this project.

Figure 1:

<img width="409" alt="s1" src="https://user-images.githubusercontent.com/89792366/236385063-65fad8ad-bdf1-413c-8505-7f010733d987.png">

Figure 1: The plot shows that Time is distributed with 2 bell curves with some skewness. This is dealt by scaling time with RobustScaler.



Figure 2:

<img width="406" alt="s2" src="https://user-images.githubusercontent.com/89792366/236385078-a38adbda-c064-4cc0-b5ff-21106579eaa4.png">

Figure 2: The plot shows that the transactions increase after 10 AM with uniform distribution but stays low during early morning.



Figure 3:

<img width="292" alt="s3" src="https://user-images.githubusercontent.com/89792366/236385116-7b112cb3-a139-4353-8d74-33668fbbae6a.png">

Figure 3: The plots show that features V10, V12, V14 have outliers, which can affect the accuracy of the models. Thus, the cut off was decided using IQR range to discard the outliers.



Figure 4:

<img width="823" alt="s4" src="https://user-images.githubusercontent.com/89792366/236385137-0269f3ea-c295-4d5e-a903-f52ea27d3f6d.png">

Figure 4: The t-sne plot shows that it is successful in differentiating between Class variable and make clusters accordingly. Thus, models should be able to predict the test dataset accurately.
 
 
### Problem Formulation

* Data:
  * Input: The input is the different datasets explained in Preprocessing/Cleanup
  * Output: Class 
  
  * Models
  
    * XGBClassifier: Imbalanced datasets are a common challenge in credit fraud detection, where the number of fraudulent transactions is typically a small proportion of the overall dataset. XGBClassifier has built-in mechanisms to handle such datasets and can adjust the weights of samples based on their relative importance, ensuring that the model is not biased towards the majority class.
    
    * LogisticRegression: Logistic regression models are highly interpretable, meaning that they provide insights into the factors that contribute to the probability of a transaction being fraudulent. The coefficients of the model can be used to identify which variables are most important in predicting fraud, allowing analysts to make informed decisions based on these insights.

### Training
* Software used:
   * Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, joblib
   
* XGBClassifier Model:

  The Undersample Data model was created and trained as follows: 
  
  
    <img width="823" alt="s5" src="https://user-images.githubusercontent.com/89792366/236386741-1794c10a-a9a5-4351-8bc3-a1dadad497c9.png">


  The confusion matrix plot:  
  
  
   <img width="722" alt="s6" src="https://user-images.githubusercontent.com/89792366/236388803-006954c2-6f6e-404b-8a23-9253a0702693.png"> 
  
  
  The plot for XGB Under Sampled Data shows good classification with 0.945 f1 score and few false predictions. This score tells us that the model might be overfitting. 
  
  
  The Oversample Data model was created and trained as follows: 
  
  
    <img width="821" alt="s7" src="https://user-images.githubusercontent.com/89792366/236389044-6ed149ea-1cd2-485d-b0b5-37429f9fa6e1.png">


  The confusion matrix plot:  
  
  
   <img width="744" alt="s8" src="https://user-images.githubusercontent.com/89792366/236389113-bb04d983-61cc-48f4-8a74-8d69d9a68e68.png"> 
  
  
  The plot for XGB OverSampled Data shows approximately perfect classification with f1 score of 0.999, which must be overfitting.


* LogisticRegression Model:


   The Undersample Data model was created and trained as follows: 
  
  
   <img width="824" alt="s9" src="https://user-images.githubusercontent.com/89792366/236389316-9135c0f1-e5af-4af9-9c5f-f843151cb0e1.png">


  The confusion matrix plot:  
  
  
   <img width="749" alt="s10" src="https://user-images.githubusercontent.com/89792366/236389439-f3d54909-730a-4b91-b91e-8ea1a9e74b76.png"> 
  
  
  The plot for LogisticRegression UnderSampled Data shows good classification but still has some false predictions with f1 score 0.939.
  
  
  The Oversample Data model was created and trained as follows: 
  
  
    <img width="824" alt="s11" src="https://user-images.githubusercontent.com/89792366/236389665-1a7a5644-439b-4297-a2b7-40139ab1859f.png">


  The confusion matrix plot:  
  
  
   <img width="762" alt="s12" src="https://user-images.githubusercontent.com/89792366/236389731-43f0d981-5403-4a61-b536-ec549ab94ecf.png"> 
  
  
  The plot for LogisticRegression OverSampled Data shows some variance because there are 1000-5000 data points that it could not classify properly with f1 score 0.949. This model seems less prone to overfitting. 
  

### Performance Comparison


* The performance metric is f1 score.

* Table:
   * UnderSample Models:

   <img width="141" alt="ut" src="https://user-images.githubusercontent.com/89792366/236390780-a50580fd-04fb-4f1e-9264-2e6bf069a183.png">
   
   * OverSample Models:
   
   <img width="143" alt="ot" src="https://user-images.githubusercontent.com/89792366/236390807-64d96fdd-4ed7-4ae0-9d94-7402b5cd3b69.png">


* XGBClassifier Model:

  The confusion matrix plot for UnderSample Model:  
  
  
   <img width="758" alt="s13" src="https://user-images.githubusercontent.com/89792366/236391270-ace61369-5b5b-4bd1-9e3a-8461b59ea151.png"> 
  
  
    The XGB Undersample Model shows that it could not classify 2300 data points properly, but overall the model has good prediction strength. 
  


  The confusion matrix plot for OverSample Model:  
  
  
  <img width="761" alt="s14" src="https://user-images.githubusercontent.com/89792366/236391627-503b7d37-0568-467f-a65c-9c100318fcf9.png">
  
  
   The XGB Oversample Model shows that it perfectly classifies the imbalanced dataset. This is an indication of overfitting model. 
  
  
  
* LogisticRegresssion model:

  The confusion matrix plot for UnderSample Model:  
  
  
   <img width="725" alt="s15" src="https://user-images.githubusercontent.com/89792366/236391789-d626f9f0-58f3-430f-910e-f3e900d8dc1f.png"> 
  
  
  The LR Undersample Model shows that it could not classify 2200 data points properly, but overall it has good prediction strength. 


  The confusion matrix plot for OverSample Model:  
  
  
   <img width="728" alt="s16" src="https://user-images.githubusercontent.com/89792366/236392263-43ebf523-3aaa-4aff-9433-af0053bc2ad0.png"> 
  
  
  The LR Oversample model shows good results with 0.0995 f1 score but could not classify 1300 points properly. 
  
  


### Conclusions

*  From the plots it is seen that the UnderSample models were better in classifying the fraud feature, which is the most important to detect. Because high classification of fraud as non-fraud would be detrimental for the company. On the other hand, OverSample models were better in classifying non-fraud feature. 


### Future Work

* In future, we will look into UnderSample models deeper because the number of non-fraud transcations classified as frauds were high, which can affect the customers. Their card can be blocked under such circumstances. Also, more preprocessing should be done to avoid outliers in Oversample dataset to affect the predictive strength.

### How to reproduce results


* To reproduce the results:

  * import XGBClassifier, LogisticRegression.
   
    * The followings commands can be used to import:
      * from xgboost import XGBClassifier
      * from sklearn.metrics import f1_score, confusion_matrix
      * from sklearn.model_selection import train_test_split
      * from sklearn.linear_model import LogisticRegression
     
   * Create the train, valid, test dataset as described:
   
   
   <img width="818" alt="s17" src="https://user-images.githubusercontent.com/89792366/236395598-bfed5350-e66b-45e0-8f4a-14ac76303289.png">
   

   * Create model as described in Training Section.
   
   * Train the model as described in Training Section.
   
   * The predictions can be made as follows for test data to get the confusion matrix.
   
    * Confusion Matrix Plot:
    
      <img width="438" alt="s18" src="https://user-images.githubusercontent.com/89792366/236395862-7f24c095-d134-4ce3-9035-349ad1b80001.png">
      
      <img width="777" alt="s19" src="https://user-images.githubusercontent.com/89792366/236395950-ba08fbdc-5dde-48a4-ba1c-775839290de6.png">


    * Repeat this method for other models.
    
    
### Overview of files in repository

  * CreditCardFraud.ipynb: It contains the entire code in one notebook.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * XGBClassifier.ipynb: Trains the first model and saves model during training.
  * Logisticregression.ipynb: Trains the second model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * model_under.joblib - file contaning XGB UnderSample Model.
  * model_over.joblib - file contaning XGB OverSample Model.
  * lr_under.joblib - file contaning LogisticRegression UnderSample Model.
  * lr_over.joblib - file contaning LogisticRegression OverSample Model.

### Software Setup

* Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, joblib
* Download seaborn in jupyter - pip install seaborn
* Download xgboost in jupyter - pip install xgboost

### Data

* Download data files required for the project from the following link:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


## Citations

* https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_1_BookContent/BookContent.html
* https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook
