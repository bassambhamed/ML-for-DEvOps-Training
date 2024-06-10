#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image


"""
This fonction serves to clean the data for trainining needed for further steps
"""
def cleanData(type):

    # Importing dataset
    dataset = pd.read_csv('datasetQWS.csv')

    # Remove duplicated and null variables
    ## duplicated data
    dataset=dataset.drop_duplicates(keep='last')

    ## Null variables
    Compliance_mean = dataset['Compliance'].mean()
    dataset['Compliance'].fillna(value = int(Compliance_mean), inplace = True )


    #Remove outliers
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
        return df

    # Apply remove outliers function
    columnsolddatset=['Availability','Throughput','Reliability','Latency']
    for i in range(0,len(columnsolddatset)):
        dataset = remove_outliers_iqr(dataset, columnsolddatset[i])
    
    #Remove non numerical data
    datasetNumerical = dataset.drop(['Service_Name','WSDL_Address'],axis=1)
    #Remove non numerical data(clustering)
    datasetNumericalclustering = dataset.drop(['WSDL_Address'],axis=1)
    ##Condition on the type of algorithm
    # get target and features (x and y)

    if(type=="Classification"):
        target=['Class']
        features=[
	    'Availability' ,
            'Throughput' ,
            'Successability' ,
            'Compliance',
            'Best_Practices'  ,
            'Latency'    ,
            'Documentation' ,
            'WsRF'     ,
            'Reliability','Response_Time']
    elif(type=="Regression"):
            target=['Reliability']
            features=['Availability' ,
                  'Throughput' ,
                  'Successability' ,
                  'Compliance',
                  'Best_Practices'  ,
                  'Latency'    ,
                  'Documentation' ,
                  'WsRF'     ,
                  'Class'   ]
    elif(type=="Clustering"):
        target=['Class']
        features=['Availability' ,
          'Throughput' ,
          'Successability' ,
          'Compliance',
          'Best_Practices'  ,
          'Latency'    ,
          'Documentation' ,
          'WsRF'     ,
          'Reliability' ,'Service_Name' ,'Response_Time' ]
        X = datasetNumericalclustering[features]
        y = datasetNumericalclustering[target]
        return X,y,features,target


    X = datasetNumerical[features]
    y = datasetNumerical[target]
    return X,y,features,target