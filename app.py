import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.tree import DecisionTreeRegressor
import os
import joblib
datapath = r"C:/Users/pavan/OneDrive/Desktop/DataScience/Assignments/Sales Project"
os.chdir(datapath)
from flask import Flask, render_template

app = Flask(__name__)

# def read_dataset():
#     sales = pd.read_excel("base_modelling_data.xlsx")
#     return sales

def read_Production_data():
    sales = pd.read_csv("bestColumns.csv")
    return sales

def preprocessing_data(sales):
    # Here i am replacing 'on trend' to 'On Trend'
    sales['Guys Segmentation'] = sales['Guys Segmentation'].replace(['On trend','On Trend'],'On Trend')
    #changing mile to min
    #sales['MSA Type'] = sales['MSA Type'].replace(['5mile','5min'],'5min')
    #sales.drop('ChangeDate',axis=1,inplace= True)
    return sales

def checking_null_values(dataset):
    column = dataset.columns
    for i in column:
        if dataset[i].isnull().any() == True:
            dataset[i]
            print(i,"---",dataset[i].isnull().sum())

def filling_null_values(dataset):
    nv = dataset.isnull().any().sum()
    print("Null values:", nv)
    print('-------------------------------------------')
    print('Null values are:')
    checking_null_values(dataset)
    print('-------------------------------------------')
    categorical = [i for i in dataset.columns if i not in dataset.describe().columns]
    #print("Categorical Data:",categorical)
    numerical = dataset.describe().columns
    #print("Numerical Data:",numerical)
    for i in categorical:
        dataset[i].fillna(dataset[i].value_counts().index[0],inplace=True)
    for i in numerical:
        dataset[i].fillna(dataset[i].median(),inplace=True)
    print("After filling null values:", dataset.isnull().any().sum())
    print('-------------------------------------------')
    return dataset

# def LabelEncoding(dataset):
#     categorical= [i for i in dataset.columns if i not in dataset.describe().columns]
#     print(categorical)
#     a=LabelEncoder()
#     for i in categorical:
#         a.fit(dataset[i])
#         b = a.transform(dataset[i])
#         dataset[i] = b
#     return dataset

# def LabelEncodingTrainingData(data):
#     dataset = data.copy()
#     save_path = 'C:\Users\pavan\OneDrive\Desktop\DataScience\Assignments\Sales Project'
#     categorical= [i for i in data.columns if i not in data.describe().columns]
#     print(categorical)
#     a=LabelEncoder()
#     for i in categorical:
#         a.fit(data[i])
#         dataset[i] = a.transform(data[i])
#         filename = f'{i}_label_encoder.pkl'
#         joblib.dump(a,f'{save_path}/{filename}')
#     return dataset

def LabelEncodingProduction(data):
    dataset = data.copy()
    categorical= [i for i in data.columns if i not in data.describe().columns]
    a=LabelEncoder()
    for i in categorical:
        filename = f'{i}_label_encoder.pkl'
        loads =  joblib.load(filename)
        dataset[i] = loads.transform(data[i])
    return dataset,data




def correlation(data):
    corr = set()
    corr_mat = data.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i,j]) > 0.8:
                col = corr_mat.columns[i]
                corr.add(col)
    data.drop(corr,axis=1)
    return data

def feature_Selection(sales):
    x = sales.drop('Sales',axis=1)
    y = sales['Sales']
    select10bestCol = SelectKBest(k=10,score_func = f_regression)
    select10bestCol.fit(x,y)
    sales10 = x.iloc[:,select10bestCol.get_support()]
    print('-------------------------------------------')
    print('Important Columns after Feature Selection: ')
    print(sales10.columns)
    return sales10


# def prep_data_for_model(data,run_method):
 
#     df = preprocessing_data(data)
#     if run_method == 'train':
#         df_final,data_original = LabelEncodingTrainingData(df)
#     else:
#         df_final,data_original = LabelEncodingProduction(df)      
#     return df_final,data_original

def makePrediction(data):
    model = joblib.load("Dt_sales_model.pkl")
    predictions = model.predict(data)
    return predictions

def save_predictions(data):
    data.to_csv("Predictions.csv", index=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])

def predict():
    main_data = read_Production_data()
    sales_data = preprocessing_data(main_data)
    sales_data =  filling_null_values(sales_data)
    sales_data,original_data = LabelEncodingProduction(sales_data)
    sales_data = correlation(sales_data)
    sales_data =  feature_Selection(sales_data)
    
    predictions = makePrediction(sales_data)
    original_data['Sales_Price_Pred'] = predictions


    save_predictions(original_data)
    return render_template("index.html",prediction_message = "Predictions made and saved to 'prediction.csv file'")

if __name__ == "__main__":
    app.run(debug=True)
    