# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model
![neural model image](https://user-images.githubusercontent.com/95342910/187116300-055fa443-9fec-4b56-b25a-b377b970b908.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
~~~
Developed by : Guru prasad.B
Reg.no : 212221230032
Program to develop a neural network regression model

### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('data').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'Input':'int','Output':'int'})

### Import the packages :

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

X = df[['Input']].values
y = df[['Output']].values
X

### Split Training and testing set :

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)

X_train1

### Model :

ai_brain = Sequential([
    Dense(4,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 4000)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[89]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)

~~~

## Dataset Information
![out1](https://user-images.githubusercontent.com/95342910/187116715-e491e305-29bb-4c20-9676-9bd6aa7f7ca8.png)

## OUTPUT
### Initiation of program
![out2](https://user-images.githubusercontent.com/95342910/187116822-edabf00f-196a-4727-a640-d3301b88473b.png)

![out3](https://user-images.githubusercontent.com/95342910/187117070-17937d57-2708-4a26-b7ca-7d0c79c389e6.png)

![out4](https://user-images.githubusercontent.com/95342910/187117145-c20e682a-8a6f-4b0c-b25c-494acc930438.png)

### Training Loss Vs Iteration Plot
![out5](https://user-images.githubusercontent.com/95342910/187117198-f1be7188-bbe7-4070-8d9b-8452884b1130.png)

![out6](https://user-images.githubusercontent.com/95342910/187117221-c41ff71b-0275-4335-9188-15c59381a03d.png)

### Test Data Root Mean Squared Error
![out7](https://user-images.githubusercontent.com/95342910/187117249-1c029965-d4ee-471f-b93a-93eb33a852aa.png)

### New Sample Data Prediction
![out8](https://user-images.githubusercontent.com/95342910/187117274-3e57d2b1-4c36-46fd-84ec-819fde675f5d.png)

## RESULT
Thus a neural network model for regression using the given dataset is written and executed successfully.
