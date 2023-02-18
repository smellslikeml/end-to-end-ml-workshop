# Databricks notebook source
# MAGIC %md
# MAGIC ## End to End ML on Databricks
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/10/Machine-Learning-graphic-1.png" width=1012/>

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

import pyspark.pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For this workshop we will use a publicly available adult dataset example found in `/databricks-datasets/`. We could also use Python or Spark to read data from databases or cloud storage.

# COMMAND ----------

# We can ls the directory and see what files we have available
dbutils.fs.ls("/databricks-datasets/adult")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Path configs

# COMMAND ----------

# Set config for database name, file paths, and table names
database_name = 'ml_income_workshop'
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')

# Paths for various Delta tables
raw_tbl_path = '/home/{}/ml_income_workshop/raw/'.format(user)
clean_tbl_path = '/home/{}/ml_income_workshop/clean/'.format(user)
inference_tbl_path = '/home/{}/ml_income_workshop/inference/'.format(user)

raw_tbl_name = 'raw_income'
clean_tbl_name = 'clean_income'
inference_tbl_name = 'inference_income'

# Delete the old database and tables if needed
spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
spark.sql('CREATE DATABASE {}'.format(database_name))
spark.sql('USE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same path variables)
dbutils.fs.rm(raw_tbl_path, recurse = True)
dbutils.fs.rm(clean_tbl_path, recurse = True)
dbutils.fs.rm(inference_tbl_path, recurse = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md #### Reading in Data

# COMMAND ----------

# MAGIC %sh cat /dbfs/databricks-datasets/adult/README.md

# COMMAND ----------

census_income_path = "/databricks-datasets/adult/adult.data"

# defining the schema for departure_delays
census_income_schema = StructType([ \
  StructField("age", IntegerType(), True), \
  StructField("workclass", StringType(), True), \
  StructField("fnlwgt", DoubleType(), True), \
  StructField("education", StringType(), True), \
  StructField("education_num", DoubleType(), True), \
  StructField("marital_status", StringType(), True), \
  StructField("occupation", StringType(), True), \
  StructField("relationship", StringType(), True), \
  StructField("race", StringType(), True), \
  StructField("sex", StringType(), True), \
  StructField("capital_gain", DoubleType(), True), \
  StructField("capital_loss", DoubleType(), True), \
  StructField("hours_per_week", DoubleType(), True), \
  StructField("native_country", StringType(), True), \
  StructField("income", StringType(), True),
])
raw_df = spark.read.schema(census_income_schema).options(header='false', delimiter=',').csv(census_income_path)

display(raw_df)

# COMMAND ----------

raw_df.write.format('delta').mode('overwrite').save(raw_tbl_path)

# COMMAND ----------

display(raw_df)

# COMMAND ----------

# MAGIC %md By clicking on the `Data Profile` tab above we can easily generate descriptive statistics on our dataset. We can also call those results using `dbutils`

# COMMAND ----------

dbutils.data.summarize(raw_df)

# COMMAND ----------

# Create table to query with SQL
spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(raw_tbl_name, raw_tbl_path)
         )

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's query our table with SQL!

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Occupations with the highest average age
# MAGIC SELECT occupation, ROUND(AVG(age)) AS avg_age
# MAGIC FROM raw_income
# MAGIC GROUP BY occupation
# MAGIC ORDER BY avg_age DESC

# COMMAND ----------

# MAGIC %md ### Data Visualization

# COMMAND ----------

# MAGIC %md We can also display the results as a table using the built-in visualizations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Occupations with the highest average age
# MAGIC SELECT occupation, ROUND(AVG(age)) AS avg_age
# MAGIC FROM raw_income
# MAGIC GROUP BY occupation
# MAGIC ORDER BY avg_age DESC

# COMMAND ----------

# MAGIC %md 
# MAGIC You can also leverage [`Pandas on Spark`](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html) to use your favorite pandas and matplotlib functions for data wrangling and visualization but with the scale and optimizations of Spark ([announcement blog](https://databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html), [docs](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html), [quickstart](https://spark.apache.org/docs/latest/api/python/getting_started/quickstart_ps.html))
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/09/Pandas-API-on-Upcoming-Apache-Spark-3.2-blog-img-4.png" width=516/>

# COMMAND ----------

# convert our raw spark distributed dataframe into a distributed pandas dataframe
raw_df_pdf = raw_df.to_pandas_on_spark()

# perform the same aggregation we did in SQL using familiar Pandas syntax
avg_age_by_occupation = raw_df_pdf.groupby("occupation").mean().round().reset_index()[["occupation", "age"]].sort_values("age", ascending = False)

# re-create the same plot using familiar pandas and matplotlib syntax distributed with Spark
avg_age_by_occupation.plot(kind = "bar", x = "occupation", y = "age")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Wrangling
# MAGIC We can leverage [`Pandas on Spark`](https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html) to clean and wrangle our data at scale. We are going to drop missing values and clean up category values.

# COMMAND ----------

# Drop missing values
clean_pdf = raw_df_pdf.dropna(axis = 0, how = 'any')

def category_cleaner(value):
  return value.strip().lower().replace('.', '').replace(',', '').replace(' ', '-')

categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']

for column in categorical_cols:
  clean_pdf[column] = clean_pdf[column].apply(lambda value: category_cleaner(value))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Engineering
# MAGIC 
# MAGIC * Bin the age into decades (min age 17 to max age 90)
# MAGIC * take the log of features with skewed distributions (capital_gain, capital_loss)

# COMMAND ----------

import numpy as np

# bin age into decades
def bin_age(value):
  if value <= 19:
    return "teens"
  elif value in range(20,30):
    return "20s"
  elif value in range(30,40):
    return "30s"
  elif value in range(40,50):
    return "40s"
  elif value in range(50,60):
    return "50s"
  elif value in range(60,100):
    return "60+"
  else:
    return "other"
  
clean_pdf['age_by_decade'] = clean_pdf['age'].apply(bin_age)
  
# Take the log of features with skewed distributions
def log_transform(value):
  return float(np.log(value + 1)) # for 0 values

clean_pdf['log_capital_gain'] = clean_pdf['capital_gain'].apply(log_transform)
clean_pdf['log_capital_loss'] = clean_pdf['capital_loss'].apply(log_transform)

# Drop columns
clean_pdf = clean_pdf.drop(['age', 'capital_gain', 'capital_loss'], axis = 1)

display(clean_pdf.head(3))

# COMMAND ----------

# we are now going to save this cleaned table as a delta file in our cloud storage and create a metadata table on top of it
clean_pdf.to_delta(clean_tbl_path)

spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(clean_tbl_name, clean_tbl_path)
         )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM clean_income
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %md
# MAGIC Let's package the all the data processing into a function for later use

# COMMAND ----------

def process_census_data(dataframe):
  """
  Function to wrap specific processing for census data tables
  Input and output is a pyspark.pandas dataframe
  """
  categorical_cols = ['workclass', 'education', 'marital_status', 
                      'occupation', 'relationship', 'race', 'sex', 
                      'native_country', 'income']
  
  # categorical column cleansing
  for column in categorical_cols:
    dataframe[column] = dataframe[column].apply(lambda value: category_cleaner(value))
  
  # bin age
  dataframe['age_by_decade'] = dataframe['age'].apply(bin_age)
  
  # log transform
  dataframe['log_capital_gain'] = dataframe['capital_gain'].apply(log_transform)
  dataframe['log_capital_loss'] = dataframe['capital_loss'].apply(log_transform)
  
  # Drop columns
  dataframe = dataframe.drop(['age', 'capital_gain', 'capital_loss'], axis = 1)
  
  return dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC Last but not least, let's create the same transformations to our inference dataset for testing later

# COMMAND ----------

census_income_test_path = "/databricks-datasets/adult/adult.test"

inference_pdf = (spark.read.schema(census_income_schema)
                          .options(header='false', delimiter=',')
                          .csv(census_income_test_path)
                          .to_pandas_on_spark()
               )

inference_pdf = process_census_data(inference_pdf)
inference_pdf.to_delta(inference_tbl_path)

spark.sql('''
             CREATE TABLE {0}
             USING DELTA 
             LOCATION '{1}'
          '''.format(inference_tbl_name, inference_tbl_path)
         )

# COMMAND ----------

# MAGIC %md
# MAGIC Great! Our dataset is ready for us to use with AutoML to train a benchmark model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/Glass-Box-Approach-to-AutoML-1-light.png" width=1012/>

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's create a glassbox AutoML model to help us automatically test different models and parameters and reduce time manually testing and tweaking ML models. We can run AutoML via the `databricks.automl` library or via the UI by creating a new [mlflow automl experiment](#mlflow/experiments).
# MAGIC 
# MAGIC Here, we'll run AutoML in the next cell.

# COMMAND ----------

import databricks.automl

summary = databricks.automl.classify(clean_pdf, target_col='income', primary_metric="f1", data_dir='dbfs:/automl/ml_income_workshop', timeout_minutes=5)

# COMMAND ----------

# MAGIC %md
# MAGIC Check out the screenshots below that walk through this process via the UI.
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2021/05/Graphic-2.png" width=708/> <img src="https://databricks.com/wp-content/uploads/2021/05/Graphic-3.png" width=708/> <img src="https://databricks.com/wp-content/uploads/2021/05/XG-Boost-1.png" width=708/>
