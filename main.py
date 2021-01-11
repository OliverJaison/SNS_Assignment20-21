
from selenium import webdriver
from bs4 import BeautifulSoup as bs4
import pandas as pd
import requests
import os

working_dir = os.getcwd()
print(working_dir)

positive_cases_csv_URL = "https://coronavirus.data.gov.uk/api/v1/data?filters=areaType=overview&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22areaCode%22:%22areaCode%22,%22date%22:%22date%22,%22newCasesBySpecimenDate%22:%22newCasesBySpecimenDate%22,%22cumCasesBySpecimenDate%22:%22cumCasesBySpecimenDate%22%7D&format=csv" 
req = requests.get(positive_cases_csv_URL)
URL_content = req.content
positive_cases_file = open("positive_cases.csv", "wb")
positive_cases_file.write(URL_content)
positive_cases_file.close()

pos_cases_df = pd.read_csv(os.path.join(working_dir, "positive_cases.csv"))
#  Columns 0, 1 and 2 are not useful as I am looking at the UK exclusively right now
pos_cases_df = pos_cases_df.drop(["areaType", "areaName", "areaCode"], axis=1)
print(pos_cases_df.head())
