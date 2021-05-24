# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load Data
data = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\Projects\\ApiPython_WineClassify\\WineClassify\\Data\\wine_dataset.csv')
print(data)

# Data Preparation
data['style'] = data['style'].replace('red',0)
data['style'] = data['style'].replace('white',1)
print(data)

print(data.isnull().sum())

# Separate the variables between predictors and target
X = data.drop('style',axis=1)
y = data['style']

print(X)
print(y)

# Create the training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=2)

print(X_train.shape)

# Training Model
model = ExtraTreesClassifier()
model.fit(X_train,y_train)

# Performance Metrics 
result = model.score(X_test,y_test)
print("Accuracy is:",result)

print(y_test[400:403])
print(X_test[400:403])

y_predx = model.predict(X_test[400:403])
print(y_predx)

y_pred = model.predict(X_test)
print(y_pred)

# Confusion Matrix
true_labels = y_test
pred_labels = y_pred

confusion_matrix = confusion_matrix(true_labels,pred_labels)
sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
ticks = np.arange(2)
plt.xticks = (ticks,ticks)
plt.yticks = (ticks, ticks)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()

# Create file with model
joblib.dump(model,'wine.pkl')

model_wine = joblib.load('wine.pkl')