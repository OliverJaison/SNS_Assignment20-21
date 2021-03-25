
from sklearn import preprocessing
import pandas as pd
import numpy as np
import requests
import os
import csv
import keras
import tabulate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy 


# Pulls data from websites and stores them in csv files

# In[2]:


def update_data():
    positive_cases_csv_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    req = requests.get(positive_cases_csv_URL)
    URL_content = req.content
    positive_cases_file = open("positive_cases.csv", "wb")
    positive_cases_file.write(URL_content)
    positive_cases_file.close()


# This is purely for the sources of data that include data outside the UK as having international data would mean too much to parse through

# In[3]:


def filter_data(filename):
    uk = list()
    with open(filename, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            if row[0] == "GBR" or row[0] == "iso_code":
                uk.append(row)
    
    with open(filename, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(uk)  


# This will only take columns in the data frame with no NaNs

# In[6]:


def no_NaNs(df):
    data = []
    for column in df.columns:
        temp = [float(i) for i in df[column]]
        if np.isnan(np.sum(np.array(temp))):
            df.drop([column], inplace=True, axis=1)


# This will only take columns that don't have a single repeating entry

# In[7]:


def no_repeat(df):
    data = []
    for column in df.columns:
        if df[column].nunique() == 1:
            df.drop([column], inplace=True, axis=1)


# This will create the input array for the Neural Network

# In[8]:


def create_features(df, window_size, forecast):
    features = []
    labels = []
    i = window_size;
    while i < len(df.iloc[:, 0]) - forecast-1:
        window = df.iloc[i - window_size: i, 1:]
        window = np.array(window)
        window = window.flatten()
        labels.append(df.iloc[i + forecast, 0])
        features.append(window)
        i += 1
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


# In[9]:


update_data() 


# In[10]:


filter_data("positive_cases.csv")


# Grabs the current working directory where the csv files are stored

# In[11]:


working_dir = os.getcwd()


# Reads the csv files into their respective dataframes

# In[12]:


pos_cases_df = pd.read_csv(os.path.join(working_dir, "positive_cases.csv"))
pos_cases_df.drop(pos_cases_df.iloc[:, 0:3], inplace = True, axis=1)
dates = pos_cases_df["date"]
pos_cases_df.drop(["date"], inplace = True, axis=1)
pos_cases_df.drop(["tests_units"], inplace = True, axis=1)


# In[13]:


no_NaNs(pos_cases_df)


# In[14]:


no_repeat(pos_cases_df)


# Code just adds a column to the dataframe that counts the number of days since the earliest data entry

# In[15]:


daysSince = []
for i in range(len(pos_cases_df)):
    daysSince.append(i)
pos_cases_df["daysSince"] = daysSince


# Here I make the feature and label matrices with a 7 day window and a 1 day forecast

# In[16]:


window_size = 7
forecast = 1
features1, labels1 = create_features(pos_cases_df, window_size, forecast)


# Splitting the feature and label matrices into training and testing by 8:2. I will split the training further down into 8:2 again for validation.

# In[17]:


training_features1, testing_features1 = np.split(features1, [int(0.8*len(features1))])
training_labels1, testing_labels1 = np.split(labels1, [int(0.8*len(labels1))])


# In[18]:


input_shape = training_features1.shape[1]


# In[19]:


NN_model = Sequential()
# Input layer
NN_model.add(Dense(128, input_dim=input_shape, kernel_initializer="normal", activation="relu"))
# Hidden Layers
NN_model.add(Dense(256, kernel_initializer="normal", activation="relu"))
NN_model.add(Dense(256, kernel_initializer="normal", activation="relu"))
NN_model.add(Dense(256, kernel_initializer="normal", activation="relu"))
# Output layer 
NN_model.add(Dense(1, kernel_initializer="normal", activation="linear"))

NN_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["accuracy"])
NN_model.summary()


# In[20]:


history1 = NN_model.fit(training_features1, training_labels1, epochs=500, validation_split=0.2)


# In[22]:


epochs = range(1, len(history1.epoch) + 1)
# plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history1.history['accuracy'], label='Training Accuracy')
# plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history1.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Training for 1 day forecast")
plt.legend()
plt.show()


# In[23]:


x = [i for i in range(int(testing_features1[0,-1]+1), int(testing_features1[-1,-1]+2))]
y = testing_labels1
x_pred = testing_features1
y_pred = NN_model.predict(x_pred)
err1 = np.abs(y - y_pred)
acc1 = np.ones(len(y)) - np.divide(err1, y)
acc_bar1 = np.mean(acc1)
rms1 = np.sqrt(np.mean((err1)**2))
print("Average accuracy:", acc_bar1)
print("RMS:", rms1)


# In[25]:


plt.plot(x, y, label="True Values")
plt.plot(x, y_pred, label="Predicted Values")
plt.xlabel("Number of Days Since 31/01/2020")
plt.ylabel("Total number of cases in the UK")
plt.title("1 Day forecast")
plt.legend()
plt.show()


# Here I make a new feature and label matrix with a 7 day window and a 2 day forecast

# In[70]:


window_size = 7
forecast = 2
features2, labels2 = create_features(pos_cases_df, window_size, forecast)


# I split the matrices down into training and testing using a 8:2 splitting

# In[71]:


training_features2, testing_features2 = np.split(features2, [int(0.8*len(features2))])
training_labels2, testing_labels2 = np.split(labels2, [int(0.8*len(labels2))])


# The model has to be trained again with this new data

# In[72]:


history2 = NN_model.fit(training_features2, training_labels2, epochs=500, validation_split=0.2)


# In[73]:


epochs = range(1, len(history2.epoch) + 1)
# plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history2.history['accuracy'], label='Training Accuracy')
# plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title("Training for 2 day forecast")
plt.legend()
plt.show()


# In[74]:


x = [i for i in range(int(testing_features2[0,-1]+1), int(testing_features2[-1,-1]+2))]
y = testing_labels2
x_pred = testing_features2
y_pred = NN_model.predict(x_pred) 
err2 = np.abs(y - y_pred)
acc2 = np.ones(len(y)) - np.divide(err2, y)
acc_bar2 = np.mean(acc2)
rms2 = np.sqrt(np.mean((err2)**2))
print("Average accuracy:", acc_bar2)
print("RMS:", rms2)


# In[31]:


plt.plot(x, y, label="True Values")
plt.plot(x, y_pred, label="Predicted Values")
plt.xlabel("Days Since first case entry")
plt.ylabel("Total number of cases in the UK")
plt.title("2 Day forecast")
plt.legend()
plt.show()


# In[50]:


window_size = 7
forecast = 5
features3, labels3 = create_features(pos_cases_df, window_size, forecast)


# In[51]:


training_features3, testing_features3 = np.split(features3, [int(0.8*len(features3))])
training_labels3, testing_labels3 = np.split(labels3, [int(0.8*len(labels3))])


# In[52]:


history3 = NN_model.fit(training_features3, training_labels3, epochs=500, validation_split=0.2)


# In[54]:


epochs = range(1, len(history3.epoch) + 1)
# plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history3.history['accuracy'], label='Training Accuracy')
# plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history3.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title("Training for 5 day forecast")
plt.legend()
plt.show()


# In[55]:


x = [i for i in range(int(testing_features3[0,-1]+1), int(testing_features3[-1,-1]+2))]
y = testing_labels3
x_pred = testing_features3
y_pred = NN_model.predict(x_pred) 
err3 = np.abs(y - y_pred)
acc3 = np.ones(len(y)) - np.divide(err3, y)
acc_bar3 = np.mean(acc3)
rms3 = np.sqrt(np.mean((err3)**2))
print("Average accuracy:", acc_bar3)
print("RMS:", rms3)


# In[60]:


plt.plot(x, y, label="True Values")
plt.plot(x, y_pred, label="Predicted Values")
plt.xlabel("Days Since first case entry")
plt.ylabel("Total number of cases in the UK")
plt.title("5 Day forecast")
plt.legend()
plt.show()


# In[64]:


window_size = 7
forecast = 10
features10, labels10 = create_features(pos_cases_df, window_size, forecast)


# In[65]:


training_features10, testing_features10 = np.split(features10, [int(0.8*len(features10))])
training_labels10, testing_labels10 = np.split(labels10, [int(0.8*len(labels10))])


# In[66]:


history10 = NN_model.fit(training_features10, training_labels10, epochs=500, validation_split=0.2)


# In[67]:


epochs = range(1, len(history10.epoch) + 1)
# plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history10.history['accuracy'], label='Training Accuracy')
# plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history10.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title("Training for 10 day forecast")
plt.legend()
plt.show()


# In[68]:


x = [i for i in range(int(testing_features10[0,-1]+1), int(testing_features10[-1,-1]+2))]
y = testing_labels10
x_pred = testing_features10
y_pred = NN_model.predict(x_pred) 
err10 = np.abs(y - y_pred)
acc10 = np.ones(len(y)) - np.divide(err10, y)
acc_bar10 = np.mean(acc10)
rms10 = np.sqrt(np.mean((err10)**2))
print("Average accuracy:", acc_bar10)
print("RMS:", rms10)


# In[69]:


plt.plot(x, y, label="True Values")
plt.plot(x, y_pred, label="Predicted Values")
plt.xlabel("Days Since first case entry")
plt.ylabel("Total number of cases in the UK")
plt.title("10 Day forecast")
plt.legend()
plt.show()


# In[44]:


window_size = 7
forecast = 50
features50, labels50 = create_features(pos_cases_df, window_size, forecast)


# In[45]:


training_features50, testing_features50 = np.split(features50, [int(0.8*len(features50))])
training_labels50, testing_labels50 = np.split(labels50, [int(0.8*len(labels50))])


# In[46]:


history50 = NN_model.fit(training_features50, training_labels50, epochs=500, validation_split=0.2)


# In[47]:


epochs = range(1, len(history50.epoch) + 1)
# plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history50.history['accuracy'], label='Training Accuracy')
# plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history50.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


# In[48]:


x = [i for i in range(int(testing_features50[0,-1]+1), int(testing_features50[-1,-1]+2))]
y = testing_labels50
x_pred = testing_features50
y_pred = NN_model.predict(x_pred) 
err50 = np.abs(y - y_pred)
acc50 = np.ones(len(y)) - np.divide(err50, y)
acc_bar50 = np.mean(acc50)
rms50 = np.sqrt(np.mean((err50)**2))
print("Average accuracy:", acc_bar50)
print("RMS:", rms50)


# In[49]:


plt.plot(x, y, label="True Values")
plt.plot(x, y_pred, label="Predicted Values")
plt.xlabel("Days Since first case entry")
plt.ylabel("Total number of cases in the UK")
plt.title("50 Day forecast")
plt.legend()
plt.show()


# In[ ]:




