from pymongo.mongo_client import MongoClient
import pandas as pd 
import json
# uniform resource identifier

from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://Awadhesh:<password>@atlascluster.tvmskyi.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)



#create  database name and collection name 
DATABASE_NAME="customer_behviour"
COLLECTION_NAME="customer_behviour_analysis"

#read the data as dataframe 
df=pd.read_csv(r"C:\Users\Avdesh\Documents\Data_science\ML\customer_project\notebook\data\my_data.csv")
df=df.drop("Unnamed:0",axis=1)

#covert the data into jason 
json_record=list(json.loads(df.T.to_json()).values())

#now dumb the data into the dataset 
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)