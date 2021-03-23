# SNS Assignment 20-21 Report
# Covid Forecasting Engine

## Table of Contents

- [Description](#Description)
- [The Data](#Data)

## Description
The problem presented is to design and build a Covid forecasting engine using a neural network. The metric to be forecasted here is the total number of positive cases recorded in the UK. That is to say, the cumulative number of cases in the UK since January 31st of 2020. The solution stated in this report involves building a deep neural network in ```Python``` using ```tensorflow``` modules and training the model using a section of the data from a regularly updated csv file. The performance of the model is then tested by inputting the previous seven days worth of data, predicting the total number of positive cases the next day and comparing the prediction with the true value taken from the csv file. 

In the following sections of this report, the choice for the source of data will be justified and and any preprocessing involved will be explained. The design of the deep neural network will also be explained and the results will be presented.

## Data
The csv file is taken from [https://covid.ourworldindata.org/data/owid-covid-data.csv](https://covid.ourworldindata.org/data/owid-covid-data.csv). Initially, the data was taken from the [United Kingdom government website](https://coronavirus.data.gov.uk/api/v1/data?filters=areaType=overview&structure=%7B%22areaType%22:%22areaType%22,%22areaName%22:%22areaName%22,%22areaCode%22:%22areaCode%22,%22date%22:%22date%22,%22newCasesByPublishDate%22:%22newCasesByPublishDate%22,%22cumCasesByPublishDate%22:%22cumCasesByPublishDate%22%7D&format=csv). The difference between the two sources of data is that the former has a significantly larger set of parameters that could perhaps help better train the neural network model. The UK goverment website has only a set of 6 parameters per day one of which is the metric to be predicted. The "ourworldindata" website has over 50 parameters per day. While it is not necessary to use all of these parameters, should the performance of the neural network fall below reasonable accuracy then giving more parameters to learn from can maybe improve results.

Through the use of the ```Requests``` module, a function was made to download the contents csv file from the website and write them into a csv file named *positive_cases.csv*.
```bash
    def update_data():
        positive_cases_csv_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        req = requests.get(positive_cases_csv_URL)
        URL_content = req.content
        positive_cases_file = open("positive_cases.csv", "wb")
        positive_cases_file.write(URL_content)
        positive_cases_file.close()
```