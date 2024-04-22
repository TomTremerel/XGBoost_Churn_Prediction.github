Hi there 

The goal of this work is to predict customer churn on an e-commerce platform. The prediction is based on variables such as the type of device used, preferred payment method, and satisfaction score.

Initially, I cleaned the data by replacing all missing values with the median of each relevant column.

Next, I conducted exploratory data analysis (EDA) to understand the relationships and distributions among all the variables, and then removed less important variables.

To improve the readability of the graphs, I grouped the variables into different categories based on whether they had more than 20 different values or fewer than 8 different values.
![distrib](https://github.com/TomTremerel/XGBoost_Churn_Prediction.github/assets/156415815/fb89071f-549d-436f-93bf-50527ca75dbd)


This graph illustrates the different distributions among certain variables relevant to explaining customer churn.

Then, I plotted the relationships between churn and these variables:

![link](https://github.com/TomTremerel/XGBoost_Churn_Prediction.github/assets/156415815/fecfc0c5-8bad-4d84-9e22-8a1bb57ffbc3)

As we can see, one of the most explanatory factors is not the satisfaction rate, contrary to what we might have expected.

The third part is the building of the model. The specifity here is that I learned how to use the sci-kit learn pipeline method which allows to implement a sequence of data transformers. Then I used a supervised model classifier : XGBoostClassifier. 

This model begins with building a decision tree and then enhances itself by incorporating other models such as Random Forest and gradient optimization.

We have an accuracy rate of 97% on the test set. 
