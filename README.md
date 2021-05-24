# Python API to Classify Red and White Wines

Let's talk about good things, let's talk about wine!

**_Wine Classify_**

Before creating the API, a wine classification model was created by training the ExtraTreesClassifier algorithm to be able to recognize the type of wine when receiving some characteristics of the wine to the classifier.

**_Let's start the project!_**

Firstly I imported the Pandas library to read my dataset using the function _.read_csv_ and help me make a data preparation.

```
      fixed_acidity  volatile_acidity  citric_acid  residual_sugar  chlorides  ...    pH  sulphates  alcohol  quality  style
0               7.4              0.70         0.00             1.9      0.076  ...  3.51       0.56      9.4        5    red
1               7.8              0.88         0.00             2.6      0.098  ...  3.20       0.68      9.8        5    red
2               7.8              0.76         0.04             2.3      0.092  ...  3.26       0.65      9.8        5    red
3              11.2              0.28         0.56             1.9      0.075  ...  3.16       0.58      9.8        6    red
4               7.4              0.70         0.00             1.9      0.076  ...  3.51       0.56      9.4        5    red
...             ...               ...          ...             ...        ...  ...   ...        ...      ...      ...    ...
6492            6.2              0.21         0.29             1.6      0.039  ...  3.27       0.50     11.2        6  white
6493            6.6              0.32         0.36             8.0      0.047  ...  3.15       0.46      9.6        5  white
6494            6.5              0.24         0.19             1.2      0.041  ...  2.99       0.46      9.4        6  white
6495            5.5              0.29         0.30             1.1      0.022  ...  3.34       0.38     12.8        7  white
6496            6.0              0.21         0.38             0.8      0.020  ...  3.26       0.32     11.8        6  white

[6497 rows x 13 columns]
```

If you prefer, use the function _.head_ to check the first five rows of the dataset.

The column of interest is response variable (target) “style”. However this column is categorical and to work with Machine Learning it is needs to transform the categorical variables in numeric variables. For such I used the function “replace”, where I switch 0 to Red and 1 to White.
```
data['style'] = data['style'].replace('red',0)
data['style'] = data['style'].replace('white',1)
```

After that I check if there are no null values with the function _.isnull()_ and your sum, with the function _.sum()_. There are no null values.
```
fixed_acidity           0
volatile_acidity        0
citric_acid             0
residual_sugar          0
chlorides               0
free_sulfur_dioxide     0
total_sulfur_dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
style                   0
dtype: int64
```
Next step was separate the variables between predictors and target.
```
X = data.drop('style',axis=1)
y = data['style']
```
Now it's time to apply the method _train_test_split()_ to avoid overfitting, where the algorithm is very adjusted in the training base.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=2)
```
In this case I chose to separate 70% of the database for training and 30% for testing. The test_size argument determines the percentage I want in my training data and the _random_state_ parameter separates the same lines for the training and test set.

I can see the size of X_train now to check if was apply correctly 70% for train.
```
(4547, 12)
```
The data in the training base is used to train the model, while the data in the test base is used to test the performance of the model, outside the sample (with new data).

With the data prepared I created the training model. 

I used the Extra Tree Classifier for this problem. It is a algorithm with a fast learning process:
```
model = ExtraTreesClassifier()
```
And with the fit method I present the data to the model:
```
model.fit(X_train,y_train)
```
Now it's time to check performance metrics. I applied Accuracy which is a fraction of correct predictions that were made in the test set.

The Acuracy was higher: 0.99. This may sound good but it may not be a good result. For unbalanced data sets (as in this case), the accuracy alone is not sufficient. Because it will not necessarily indicate how good our classifier is.

A test can be done to choose some rows from y_test and predict the same rows in X_test.
```
y_pred = model.predict(X_test['add here the number of rows chosen’])
```
The Confusion Matrix can be used as a complement, where each row represents the real class and each column represents the predicted class.

With the Confusion Matrix, it can be seen that the classifier used was good, since predictions were predominantly correct.

![](\WineClassify\Chart\ConfusionMatrix.png)

**_API_**

An API was created and tests were performed on Postman, as follows:

![](\Api\Postman.png)

The data was taken from Kaggle. Follow the link: https://www.kaggle.com/dell4010/wine-dataset							