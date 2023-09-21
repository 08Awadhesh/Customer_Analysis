import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass
import pandas as pd # Python library for data analysis and data frame
import numpy as np # Numerical Python library for linear algebra and computations
pd.set_option('display.max_columns', None) # code to display all columns

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import date, datetime # for manupulating time and date columns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler # For handling categorical column and scaling numeric columns

# Libraries for clustering and evaluation
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore") # To prevent kernel from showing any warning

# set the color palette
palette = sns.color_palette(["#292859",'#373482','#504DB6','#5B59DD'])
sns.palplot(palette) # print color palette




@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)
    
        

class DataIngestion:
    def __init__(self):
        
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()


    def export_collection_as_dataframe(self,collection_name, db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_ID" in df.columns.to_list():
                df = df.drop(columns=["ID"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            # Creating Age and Years_Customer ( Amount of years a personn has been customer) columns.
            df['Age'] = (df["Dt_Customer"].dt.year.max()) - (df['Year_Birth'].dt.year)
            df['Years_Customer'] = (df["Dt_Customer"].dt.year.max()) - (df['Dt_Customer'].dt.year)
            df['Days_Customer'] = (df["Dt_Customer"].max()) - (df['Dt_Customer'])

            # Total amount spent on products
            df['TotalMntSpent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProducts']

            # Total number of purchases made
            df['TotalNumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

            # Total number of accepted campaigns
            df['Total_Acc_Cmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']

            # adding columns about the day, month and year cutomer joined
            df['Year_Joined'] = df['Dt_Customer'].dt.year
            df['Month_Joined'] = df['Dt_Customer'].dt.strftime("%B")
            df['Day_Joined'] = df['Dt_Customer'].dt.day_name()
            # dividing age into groups
            df['Age_Group'] = pd.cut(x = df['Age'], bins = [17, 24, 44, 64, 150],
                        labels = ['Young adult','Adult','Middel Aged','Senior Citizen'])
            # Total children living in the household
            df["Children"] = df["Kidhome"] +  df["Teenhome"]

             #Deriving living situation by marital status
            df["Partner"]=df["Marital_Status"].replace({"Married":"Yes", "Together":"Yes", "Absurd":"No", "Widow":"No", "YOLO":"No", "Divorced":"No", "Single":"No","Alone":"No"})

           #Segmenting education levels in three groups
            df["Education_Level"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
            # Dropping useless columns
            df.drop(['ID','Z_CostContact','Z_Revenue','Year_Birth','Dt_Customer'], axis=1, inplace=True) 
            # Converting Days_Joined to int format
            df['Days_Customer'] = df['Days_Customer'].dt.days.astype('int16')
            df1 = df.copy() # make a copy
            
            df1.drop(['Education','Marital_Status','Years_Customer','Year_Joined','Month_Joined','Day_Joined'], axis=1, inplace=True)
            num_col = df1.select_dtypes(include = np.number).columns
            for col in num_col:
                q1 = df1[col].quantile(0.25)
                q3 = df1[col].quantile(0.75)
                iqr = q3-q1
                ll = q1-(1.5*iqr)
                ul = q3+(1.5*iqr)
                for ind in df1[col].index:
                 if df1.loc[ind,col]>ul:
                   df1.loc[ind,col]=ul
                 elif df1.loc[ind,col]<ll:
                   df1.loc[ind,col]=ll
                 else:
                      pass
                print("Outliers have been taken care of")
            return df1

        except Exception as e:
            raise CustomException(e, sys)

        
    def export_data_into_feature_store_file_path(self)->pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method reads data from mongodb and saves it into artifacts. 
        
        Output      :   dataset is returned as a pd.DataFrame
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
       
        """
        try:
            logging.info(f"Exporting data from mongodb")
            raw_file_path  = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path,exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                                                              collection_name= MONGO_COLLECTION_NAME,
                                                              db_name = MONGO_DATABASE_NAME)
            

            logging.info(f"Saving exported data into feature store file path: {raw_file_path}")
        
            feature_store_file_path = os.path.join(raw_file_path,'my_data.csv')
            sensor_data.to_csv(feature_store_file_path,index=False)
           

            return feature_store_file_path
            

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self) -> Path:
        """
            Method Name :   initiate_data_ingestion
            Description :   This method initiates the data ingestion components of training pipeline 
            
            Output      :   train set and test set are returned as the artifacts of data ingestion components
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("Got the data from mongodb")


            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            
            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys) from e