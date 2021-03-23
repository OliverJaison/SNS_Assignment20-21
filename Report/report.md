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
This newly written csv file will have Covid data from all over the world. However, for the purposes of this problem, only the data from the UK is needed. Using the ```csv``` module, a function was made to remove all non-UK data from the csv file directly. This was chosen to be done over simply removing the data from the dataframe that the csv is imported into because it made debugging the dataframe and the written functions significantly easier than having to reload the csv each runtime. 
```bash
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
```
Now that the csv file only had data localized to the UK, the ```Pandas``` and ```os``` modules are used to import the csv into a dataframe. After this, the first three columns are dropped because the information contained in them is a repeating entry for location and iso code. The date column is also dropped because it is not needed for the neural network input. The dates column is replaced and will be discuss later. The code also shows that a column called *tests_units* is removed. This is because every entry is just *tests_units* and these entries cannot be ennumerated. 
```bash
working_dir = os.getcwd()
pos_cases_df = pd.read_csv(os.path.join(working_dir, "positive_cases.csv"))
pos_cases_df.drop(pos_cases_df.iloc[:, 0:3], inplace = True, axis=1)
dates = pos_cases_df["date"]
pos_cases_df.drop(["date"], inplace = True, axis=1)
pos_cases_df.drop(["tests_units"], inplace = True, axis=1)
```
