# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
<div style="font-size: 40px">
<b>
A neural network is a computational model inspired by the structure and function of the human brain. It consists of interconnected nodes, or neurons, organized into layers. Information flows through these neurons, with each neuron receiving input, processing it, and passing the output to the next layer of neurons. The connections between neurons are governed by weights, which determine the strength of the connections.<br><br>During training, the network adjusts these weights through a process called backpropagation, where it learns to minimize the difference between its predictions and the actual outputs, typically guided by a loss function.Neural networks are used for a variety of tasks, including image recognition, natural language processing, and reinforcement learning. Convolutional Neural Networks (CNNs) are particularly effective for tasks involving images, as they can automatically learn to extract features from raw pixel data.<br><br>
Recurrent Neural Networks (RNNs) are well-suited for sequential data, such as time series or natural language, due to their ability to capture temporal dependencies. Additionally, advancements like Generative Adversarial Networks (GANs) enable the generation of new data samples that closely resemble real data, which has applications in image synthesis and data augmentation.
</b>
</div>

## Neural Network Model

<img src="https://github.com/Janarthanan2/DEEP_LEARNING_Ex01_basic-NN-model/assets/119393515/df634f30-710a-4f57-b054-91c74aa1ad84" width=50%>

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

```
Developed By: JANARTHANAN V K
Register Number: 212222230051
```

```python
import gspread
from google.auth import default
import pandas as pd
creds,_ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('EX01').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['input']].values
y = df[['output']].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 1, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()

loss_df = pd.DataFrame(AI_Brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
AI_Brain.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)
```

## Dataset Information

<img src="https://github.com/Janarthanan2/DEEP_LEARNING_Ex01_basic-NN-model/assets/119393515/e36cb531-627e-4688-876c-416460a2fb07" width=50%>

## OUTPUT

### Training Loss Vs Iteration Plot

<img src="https://github.com/Janarthanan2/DEEP_LEARNING_Ex01_basic-NN-model/assets/119393515/5ec9db40-05cb-4b7d-8e84-91d1fee02bce" width=35%>

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT
