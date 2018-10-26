# Predicting West Nile Virus in Chicago

## Introduction

West Nile virus is a viral infection that is typically spread by mosquitoes. The species that is both the most susceptible to spreading this disease as well as the species we will be looking at is the Culex species. In recent years, there has been an influx of cases of West Nile virus in Chicago, Illinois. Due to this epidemic, measures must be taken in order to reduce the outbreak of this serious disease. One preventative measure that can be taken is to spray the areas with larvicides and/or pesticides that have the highest concentration of mosquitoes with West Nile virus. It, however, can be difficult to predict which areas will have these high concentrations of mosquitoes.

## Purpose

Given the difficult nature of predicting West Nile virus, an evaluation of all potential areas as well as all factors that contribute to the spread of this virus could provide Chicago Public Health sectors with important information about the process of preventing West Nile virus. Classification models that predict possible areas of threat for the virus could potentially influence the methods of larvicide and pesticide spraying for years to come. **With this in mind, we aim to build a machine learning model that identifies which features are most important when determining which areas of Chicago have the most cases of West Nile virus.** Using this model, Public Health sectors can look at potential places to spray, and can focus their efforts in these locations to prevent this deadly infection.

## Data

The data we used was from the online Kaggle date source. These included train, test, weather and spray datasets. The train dataset is organized in a way that allows us to evaluate the species of mosquitoes, the location of mosquito traps, and the number of mosquitoes in each trap. The test data includes all of this except the number of mosquitoes. The weather data consists of weather condition features that contribute to the lifespan and breeding of mosquitoes. Lastly, the spray data contains all spray efforts of past years, specifically 2011 and 2013.

The shapes of each data sets are:
- Train: 10.5k x 12 / 113.63 KB
- Test: 116k x 11 / 1.4 MB
- Weather: 2945 x 22 / 74.32 KB
- Spray: 14.8k x 4 / 132.58 KB

For a more in depth description of our data, please reference [Kaggle](https://www.kaggle.com/c/predict-west-nile-virus/data).

## Methods

### [Data Cleaning](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Code/01_Data_Cleaning.ipynb)

After reading in our datasets from Kaggle, we first focused on filling the null values. Originally, all null values were represented as the string 'M.' We filled these values with NaNs because it is easier to work with NaNs rather than strings when looking at the count of null values in each row and column. The most null values we found were in the weather dataset, specifically in the values with station 2 as their location. Using a simple pandas function, we filled these nulls with the corresponding values from station 1. We felt that the values for both stations were equivalent because they had similar patterns in data. We then researched which weather conditions are important for the breeding and lifespan of mosquitoes. After analyzing the data, we found that the columns 'Water1,' 'ResultSpeed,' 'ResultDir,' 'SeaLevel,' 'AvgSpeed,' 'SnowFall,' 'Depth,' 'Depart,' 'Heat,' 'Cool' and 'StnPressure' did not give us any useful information for predicting mosquito behavior. Other columns we dropped were 'Street,' 'AddressNumberAndStreet,' 'Address,' and 'AddressAccuracy' from the train and test datasets due to the fact that these features were both uninformative and redundant. In total, we ended up dropping only columns and no rows. The shape of our data sets changed as follows:

- Train: 10.5k x 8
- Test: 116k x 7
- Weather: 2945 x 13

### [EDA](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Code/02_Data_Visualization.ipynb)
- [Heatmap](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Images/Heatmap.png)

- [Species with West Nile virus](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Images/Species.png)

### [Feature Engineering](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Code/03_Feature_Engineering_%26_Pre-processing.ipynb)

The first thing we wanted to focus on was changing the index. By changing the index to date, it is easier to reference each observation, and allows us to merge datasets by their dates. For further preprocessing, we wanted to create informative features that could be used to accurately predict West Nile virus. We found that daylight is an important indicator of mosquito behavior, specifically when they feed. We subtracted the 'Sunset' feature from the weather dataset with the 'Sunrise' feature then dropped these columns because they were no longer needed. Other important features that were created were 'Rain,' 'Fog,' and 'Humidity.' Based on outside research, we found that these factors highly predict the presence of mosquitoes. We also wanted to find the mean of the previous 7 days, or the number of days out of the previous 7 days that an event occurred. This gave us meaningful insight into the process of each trap being set up. After dummying all necessary variables, we concatenated all the datasets with the necessary features into one data frame. This data frame was then used to create X and y variables for modeling.

### [Modeling](https://git.generalassemb.ly/MatthewParker/Project-4/tree/master/Code/04_Modeling)

We trained four classification models to predict which features are most important when determining areas of West Nile virus. We used Grid Search to optimize parameters for each model, and used ROC AUC as our scoring metric to measure model strength.
The models are listed below:

#### Adaptive Boosting

This model is meant to enhance machine learning algorithms. The default for this model uses the Decision Tree Classifier, but it can also be used to enhance other models. This model performs the best on binary classification, and it works by creating a decision tree which it learns from each decision it makes. How it does this is by fitting a single decision tree for our data, and looks at the predictions that it got wrong. It then changes the data so that the observations that were incorrect have more weight for the next decision tree.

#### Gradient Boosting

This classification model uses "weak" models, and uses the residuals to then build a more robust model. It first creates a single decision tree, which is likely to be overfit to the training data, and makes predictions. The model then looks at how wrong it was, then fits another decision tree based on the residuals over and over again, combining what each tree has learned. 

#### Logistic Regression

Logistic regression performs well in binary outcomes. Because the y varaible is either 0 or 1, we have an idea that Logistic Regression will be a good choice. Also, the betas of Logistic Regression are easy to interpret compared to other models because they are the log of odds. Therefore, in order to properly interpret the coefficient of this model, we need to exponentiate them.

#### Random Forest

Random Forest is an ensemble learning method that is flexible and easy to use. It is one of the most used algorithms, because of its simplicity and the fact that it can be used for both classification and regression tasks. It is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset. It combats high variance by adding additional randomness to the model, while growing the trees. Instead of searching for the most important feature while splitting a node, it searches for the best feature among a random subset of features. This results in a wide diversity that generally results in a better model.

| Model                 | Pros   | Cons
|-------------------------|------------|----------
| Adaptive Boost         |  <ul><li>Multiple Models can be applied to AdaBoost / Versatile </li><li>Simple and easy to program</li><li> Automatically handels missing values </li><li>  Doesn't overfit easily </li><li> Learns from previous models & improves weights for the next model  </li></ul>     | <ul><li> Extremely sensitive to data that is noisy & data outliers </li></ul>
Gradient Boosting   | <ul><li>Don't need to transforms variables </li><li>Handles missing values automatically </li><li>Can approximate most nonlinear functions  </li></ul>   |  <ul><li>If you run too many iterations it can lead to an overfit model </li><li>Extremely sensitive to data that is noisy & data outliers </li><li>Tuning the right parameters is imparative </li><li>Long time to Run </li></ul>
| Random Forest           | <ul><li>Fits on multiple decision trees with a subset of features that helps to decrease variance </li><li>Don't need to transforms variables </li><li>Great usability </li><li>Handles unbalanced classes and missing values very well </li><li>Gives a great idea of feature importance in making decisions for data  </li></ul>     | <ul><li>Difficult to interpret </li><li>Not as strong of an estimator on regression when estimating values at the extremities of the distribution of response values </li></ul>
Logistic Regression         |<ul><li>The ability to use Lasso & Ridge regression for optimal feature selection </li><li>Fast to train the model </li><li>Best for predicting probabilites </li><li>You can use the output for ranking instead of classifcation </li><li>Easily interpreted    </li></ul>   | <ul><li>Can suffer from outliers</li></ul>

### [Model Evaluation](https://git.generalassemb.ly/MatthewParker/Project-4/blob/master/Code/05_Model_Evaluation.ipynb)
After looking at the metrics for each of our models, the model that would be best for predicting West Nile virus would be the Random Forest Model. The metrics that are the most important for this particular data science problem would be recall and precision. Even though Logistic Regression had a higher recall rate that Random Forest, it predicted far more cases of West Nile incorrectly, more than double, than Random Forest did. For these reason, I believe that Random Forrest is our best performing model for this specific problem moving forward.

|         Model            | Recall | Precision | Kaggle ROC AUC |
|---------------------|--------|-----------|----------------|
| Adaptive Boosting   | 29.7%  | 19.84%    | .700           |
| Gradient Boosting   | 0.6%    | 16.67%    | .709           |
| Logistic Regression | 75.8%  | 11.5%     | .743           |
| Random Forest       | 69.1%  | 21.5%     | .723           |


## Next Steps
This project was particularly interesting due to the time nature of the training and testing data. It was particularly interesting in terms of optimizing for a metric that is not accuracy, and dealing with very unbalanced classes. Moving forward, it would be interesting to have full access to the testing data to see if we would be able to predict the presence of West Nile virus for years after 2015. It would also be interesting to see if this model would be able to predict West Nile virus in other American cities as well.
