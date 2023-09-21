import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from components.data_ingestion import DataIngestion

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
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
     self.data_transformation_config=DataTransformationConfig()
     self.export_collection_as_dataframe=DataIngestion.export_collection_as_dataframe()

     def initiate_data_transformation(self):
         try:
            logging.info("Data Transformation Inintiated")
            

            subset = df1[['Income','Kidhome','Teenhome','Age','Partner','Education_Level']]
                    # I am not scaling the kidhome, teenhome cols, cause thire min, max lies between 0 & 2
            num_cols = ['Income','Age']
            numeric_pipeline = make_pipeline(StandardScaler())
            ord_cols = ['Education_Level']
            ordinal_pipeline = make_pipeline(OrdinalEncoder(categories=[['Undergraduate','Graduate','Postgraduate']])) 
            nom_cols = ['Partner']
            nominal_pipeline = make_pipeline(OneHotEncoder())
            # stack your pipelines in column transformer
            transformer = ColumnTransformer(transformers=[('num',numeric_pipeline,num_cols),
                                           ('ordinal', ordinal_pipeline,ord_cols),
                                              ('nominal' ,nominal_pipeline,nom_cols)
                                             ])


            transformed = transformer.fit_transform(subset)
            # using k-means to form clusters
            kmeans = KMeans(n_clusters=4, random_state=42)
            subset['Clusters'] = kmeans.fit_predict(transformed) 
         except Exception as e:
            logging.info('Exception occured in data Transdormation')
            raise CustomException(e,sys)

        