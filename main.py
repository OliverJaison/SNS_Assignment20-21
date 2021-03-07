from bs4 import BeautifulSoup as bs4
from datetime import date
from sklearn import preprocessing
import pandas as pd
import numpy as np
import requests
import os
import csv


# Pulls data from websites and stores them in csv files
def update_data():
    positive_cases_csv_URL1 = "https://coronavirus.data.gov.uk/api/v1/data?filters=areaType=overview&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22areaCode%22:%22areaCode%22,%22date%22:%22date%22,%22newCasesBySpecimenDate%22:%22newCasesBySpecimenDate%22,%22cumCasesBySpecimenDate%22:%22cumCasesBySpecimenDate%22%7D&format=csv" 
    positive_cases_csv_URL2 = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    req1 = requests.get(positive_cases_csv_URL1)
    req2 = requests.get(positive_cases_csv_URL2)
    URL_content1 = req1.content
    URL_content2 = req2.content
    positive_cases_file1 = open("positive_cases1.csv", "wb")
    positive_cases_file1.write(URL_content1)
    positive_cases_file1.close()
    positive_cases_file2 = open("positive_cases2.csv", "wb")
    positive_cases_file2.write(URL_content2)
    positive_cases_file2.close()


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
filter_data("positive_cases2.csv")
# Grabs the current working directory where the csv files are stored
working_dir = os.getcwd()
# Reads the csv files into their respective dataframes
pos_cases_df1 = pd.read_csv(os.path.join(working_dir, "positive_cases1.csv"))
pos_cases_df2 = pd.read_csv(os.path.join(working_dir, "positive_cases2.csv"))

# Columns 0, 1 and 2 are not useful as I am looking at the UK exclusively right now for first csv
# pos_cases_df1 = pos_cases_df1.drop(["areaType", "areaName", "areaCode"], axis=1)
# lim = numberofdays(pos_cases_df1.iloc[0][0])
# daysSince = []
# for i in range(lim, -1, -1):
#     daysSince.append(i)
# pos_cases_df1["daysSince"] = daysSince
# Above code just adds a column to the dataframe that counts the number of days since the earliest data entry

# Dropping unnecessary information for second dataframe
pos_cases_df2.drop(pos_cases_df2.iloc[:, 0:3], inplace=True, axis=1)
pos_cases_df2.drop(pos_cases_df2.iloc[:, 3:7], inplace=True, axis=1)
pos_cases_df2.drop(pos_cases_df2.iloc[:, 5:], inplace=True, axis=1)

print(pos_cases_df2.head(), "\n")
# Normalize all the data in the dataframe
normalise_dataframe(pos_cases_df2)
print(pos_cases_df2.head())