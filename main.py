from bs4 import BeautifulSoup as bs4
from datetime import date
from sklearn import preprocessing
import pandas as pd
import numpy as np
import requests
import os
import csv
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy 


# Pulls data from websites and stores them in csv files
def update_data():
    positive_cases_csv_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    req = requests.get(positive_cases_csv_URL)
    URL_content = req.content
    positive_cases_file = open("positive_cases.csv", "wb")
    positive_cases_file.write(URL_content)
    positive_cases_file.close()


# This is purely for the sources of data that include data outside the UK as having international data would mean too much to parse through
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

       
# Counts the number of days since the earliest data entry
def numberofdays(date_in_question):
    start_date = date(2020, 1, 31)
    dateq = date_in_question.split("-")
    cdate = date(int(dateq[0]), int(dateq[1]), int(dateq[2]))
    return (cdate - start_date).days


# This will normalise all the data in a dataframe
def normalise_dataframe(df):
    for i in range(1, len(df.columns)):
        maxi = max(df.iloc[:,i])
        mini = min(df.iloc[:,i])
        for j in range(len(df.iloc[:,0])):
            df.iloc[j, i] = (df.iloc[j, i] - mini)/(maxi-mini)


update_data() 
filter_data("positive_cases.csv")
# Grabs the current working directory where the csv files are stored
working_dir = os.getcwd()
# Reads the csv files into their respective dataframes
pos_cases_df = pd.read_csv(os.path.join(working_dir, "positive_cases.csv"))

# Above code just adds a column to the dataframe that counts the number of days since the earliest data entry

# Dropping unnecessary information for second dataframe
pos_cases_df.drop(pos_cases_df.iloc[:, 0:3], inplace=True, axis=1)
pos_cases_df.drop(pos_cases_df.iloc[:, 3:7], inplace=True, axis=1)
pos_cases_df.drop(pos_cases_df.iloc[:, 5:], inplace=True, axis=1)

# Add an easier metric to process the dates by counting the number of days since the earliest entry
# daysSince = []
# for i in range(len(pos_cases_df)):
#     daysSince.append(i)
# pos_cases_df["daysSince"] = daysSince

# Normalize all the data in the dataframe
normalise_dataframe(pos_cases_df)
daysSince = []
for i in range(len(pos_cases_df)):
    daysSince.append(i)
pos_cases_df["daysSince"] = daysSince
print(pos_cases_df.tail())