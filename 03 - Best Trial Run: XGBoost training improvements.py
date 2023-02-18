# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **ml-workshop-2.0** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/4212189882465177/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/4212189882465089) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _10.2.x-cpu-ml-scala2.12_

# COMMAND ----------

# MAGIC %md
# MAGIC **Be sure to update the mlflow experiment path appropriately!**

# COMMAND ----------

import mlflow
import databricks.automl_runtime

from pyspark.sql.functions import *

# Use MLflow to track experiments
mlflow.set_experiment("/Shared/ML-Workshop-2.0/End-to-End-ML/ml-workshop-income-classifier")

target_col = "income"
database_name = 'ml_income_workshop'
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load input data into a pandas DataFrame.
import pandas as pd
df_loaded = spark.table("ml_income_workshop.clean_income").toPandas()

## Data to be Scored
inference_data = spark.read.table('ml_income_workshop.inference_income').toPandas()

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["age_by_decade", "fnlwgt", "education", "occupation", "hours_per_week", "relationship", "workclass", "log_capital_gain", "log_capital_loss", "native_country"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Incorporating insights from Data Exploration Notebook
# MAGIC 
# MAGIC According to the data exploration notebook, we have a few features with high correlation. We can try dropping some of these features to reduce redundant information. We'll also drop features with little correlation to the income column.
# MAGIC 
# MAGIC In this case, we'll drop the `workclass`, `sex`, `race`, `education_num`, and `marital_status` columns

# COMMAND ----------

df_loaded.drop(["workclass", "sex", "race", "education_num", "marital_status"], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer", SimpleImputer(strategy="mean"))
])

transformers.append(("numerical", numerical_pipeline, ["fnlwgt", "hours_per_week", "log_capital_gain", "log_capital_loss"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

transformers.append(("onehot", one_hot_encoder, ["age_by_decade", "education", "occupation", "relationship", "workclass", "native_country"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Split
# MAGIC Split the input data into 2 sets:
# MAGIC - Train (80% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

# Split out train data
X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, train_size=0.8, random_state=149849802, stratify=split_y)


# COMMAND ----------

X_test = inference_data.drop([target_col], axis=1)
y_test = inference_data[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4212189882465177/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBClassifier

help(XGBClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Incorporating insights from Data Exploration: Downsampling

# COMMAND ----------

# RandomUnderSampler for class imbalance (decrease <=50K label count)
from imblearn.under_sampling import RandomUnderSampler

# From our data exploration notebook, class ratio looks like 75/25 (<=50k/>=50k)
undersampler = RandomUnderSampler(random_state=42)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from imblearn.pipeline import make_pipeline

set_config(display="diagram")

xgbc_classifier = XGBClassifier(
  colsample_bytree=0.5562503325532802,
  learning_rate=0.26571572922086373,
  max_depth=5,
  min_child_weight=5,
  n_estimators=30,
  n_jobs=100,
  subsample=0.6859242756647854,
  verbosity=0,
  random_state=149849802,
)

model = make_pipeline(col_selector, preprocessor, standardizer, undersampler, xgbc_classifier)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    
    metrics_pdf = pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"])
    metrics_pdf["dataset"] = ["ml_income_workshop.clean_income", "ml_income_workshop.inference_income"]
    metrics_df = spark.createDataFrame(metrics_pdf)
    display(metrics_df)

# COMMAND ----------

# Save metrics to a delta table
metrics_df.write.mode("overwrite").saveAsTable(f"{database_name}.metric_data_bronze")

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import xgboost
from mlflow.tracking import MlflowClient

xgbc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(xgbc_temp_dir)
xgbc_client = MlflowClient()
xgbc_model_env_path = xgbc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgbc_temp_dir)
xgbc_model_env_str = open(xgbc_model_env_path)
xgbc_parsed_model_env_str = yaml.load(xgbc_model_env_str, Loader=yaml.FullLoader)

xgbc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

with open(xgbc_model_env_path, "w") as f:
  f.write(yaml.dump(xgbc_parsed_model_env_str))
xgbc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgbc_model_env_path, artifact_path="model")
shutil.rmtree(xgbc_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    sample_size = 500
    train_sample = X_train.sample(n=sample_size)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow stats to Delta Lake Table

# COMMAND ----------

expId = mlflow.get_experiment_by_name("/Shared/ML-Workshop-2.0/End-to-End-ML/ml-workshop-income-classifier").experiment_id

mlflow_df = spark.read.format("mlflow-experiment").load(expId)

refined_mlflow_df = mlflow_df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column

# COMMAND ----------

refined_mlflow_df.write.mode("overwrite").saveAsTable(f"{database_name}.experiment_data_bronze")

# COMMAND ----------

# MAGIC %md 
# MAGIC We can also save our AutoML experiment results to a Delta Table

# COMMAND ----------

automl_mlflow = "/Users/salma.mayorquin@databricks.com/databricks_automl/22-02-20-03:37-01 - Data Preparation: Feature Engineering and AutoML-7b753624/01 - Data Preparation: Feature Engineering and AutoML-Experiment-7b753624"

automl_expId = mlflow.get_experiment_by_name(automl_mlflow).experiment_id

automl_mlflow_df = spark.read.format("mlflow-experiment").load(automl_expId)

refined_automl_mlflow_df = automl_mlflow_df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column

# COMMAND ----------

refined_automl_mlflow_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{database_name}.automl_data_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Data Drift  
# MAGIC   
# MAGIC Understanding data drift is key to understanding when it is time to retrain your model. When you train a model, you are training it on a sample of data. While these training datasets are usually quite large, they don't represent changes that may happend to the data in the future. For instance, as new US census data gets collected, new societal factors could appear in the data coming into the model to be scored that the model does not know how to properly score.  
# MAGIC   
# MAGIC Monitoring for this drift is important so that you can retrain and refresh the model to allow for the model to adapt.  
# MAGIC   
# MAGIC The short example of this that we are showing today uses the [Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) to compare the distribution of the training dataset with the incoming data that is being scored by the model.

# COMMAND ----------

# running Kolmogorov-Smirnov test for numerical columns
from scipy import stats
from pyspark.sql.types import *

from datetime import datetime

def calculate_numerical_drift(training_dataset, comparison_dataset, comparison_dataset_name, cols, p_value, date):
  drift_data = []
  for col in cols:
    passed = 1
    test = stats.ks_2samp(training_dataset[col], comparison_dataset[col])
    if test[1] < p_value:
      passed = 0
    drift_data.append((date, comparison_dataset_name, col, float(test[0]), float(test[1]), passed))
  return drift_data

# COMMAND ----------

p_value = 0.05
numerical_cols = ["fnlwgt", "hours_per_week", "log_capital_gain", "log_capital_loss"]

dataset_name = "ml_income_workshop.inference_income"
date = datetime.strptime("2000-01-01", '%Y-%m-%d').date() # simulated date for demo purpose

numerical_cols = ["fnlwgt", "hours_per_week", "log_capital_gain", "log_capital_loss"]

drift_data = calculate_numerical_drift(df_loaded, inference_data, dataset_name, numerical_cols, p_value, date)

# COMMAND ----------

driftSchema = StructType([StructField("date", DateType(), True), \
                          StructField("dataset", StringType(), True), \
                          StructField("column", StringType(), True), \
                          StructField("statistic", FloatType(), True), \
                          StructField("pvalue", FloatType(), True), \
                          StructField("passed", IntegerType(), True)\
                      ])

numerical_data_drift_df = spark.createDataFrame(data=drift_data, schema=driftSchema)
display(numerical_data_drift_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_income_workshop.numerical_drift_income

# COMMAND ----------

# Write results to a delta table for future analysis
numerical_data_drift_df.write.mode("overwrite").saveAsTable(f"{database_name}.numerical_drift_income")

# COMMAND ----------

# MAGIC %md
# MAGIC We can perturbe our inference dataset to simulate how data can change over time.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml_income_workshop.modified_inference_data

# COMMAND ----------

import random

def add_noise(value, max_noise=20):
  """
  Simulate change in distribution by adding random noise
  """
  noise = random.randint(0, max_noise)
  return value + noise

modified_inference_data = inference_data.copy()
modified_inference_data[numerical_cols] = modified_inference_data[numerical_cols].apply(add_noise, axis = 1)

modified_inference_data_df = spark.createDataFrame(modified_inference_data)

# Write for future reference
modified_inference_data_df.write.mode("overwrite").saveAsTable(f"{database_name}.modified_inference_data")
display(modified_inference_data_df)

# COMMAND ----------

date = datetime.strptime("2010-01-01", '%Y-%m-%d').date() # simulated date for demo purpose
dataset_name = "ml_income_workshop.modified_inference_income"

modified_drift_data = calculate_numerical_drift(df_loaded, modified_inference_data, dataset_name, numerical_cols, p_value, date)

modified_numerical_drift_data = spark.createDataFrame(data=modified_drift_data, schema=driftSchema)
display(modified_numerical_drift_data)

# COMMAND ----------

# append this new data to our drift table
modified_numerical_drift_data.write.format("delta").mode("append").saveAsTable("ml_income_workshop.numerical_drift_income")

# COMMAND ----------

display(spark.table("ml_income_workshop.numerical_drift_income"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can also see how our model scores on this modified data

# COMMAND ----------

X_modified = modified_inference_data.drop([target_col], axis=1)
y_modified = modified_inference_data[target_col]

# Log metrics for the modified set
xgbc_mod_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_modified, y_modified, prefix="mod_")

xgbc_mod_metrics = {k.replace("mod_", ""): v for k, v in xgbc_mod_metrics.items()}
  
mod_metrics_pdf = pd.DataFrame([xgbc_mod_metrics])
mod_metrics_pdf["dataset"] = ["ml_income_workshop.modified_inference_income"]
mod_metrics_df = spark.createDataFrame(mod_metrics_pdf)
display(mod_metrics_df)

# COMMAND ----------

# append this new data to our metrics table
mod_metrics_df.write.format("delta").mode("append").saveAsTable("ml_income_workshop.metric_data_bronze")

# COMMAND ----------

display(spark.table("ml_income_workshop.metric_data_bronze"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift Monitoring
# MAGIC 
# MAGIC From here, you can visualize and query the various tables we created from training and data metadata using [Databricks SQL](https://databricks.com/product/databricks-sql). You can trigger alerts on custom queries to notify you when you should consider retraining your model.

# COMMAND ----------

# MAGIC %md
# MAGIC [Here](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/64cbc6a0-bbd8-4612-9275-67327099a6dd-end-to-end-ml-workshop?o=1444828305810485) is our simple dashboard example
