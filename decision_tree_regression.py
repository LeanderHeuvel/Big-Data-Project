from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

seed = 284
sc = SparkContext(appName='feature_extraction')

pressure_path = "hdfs://ctit048.ewi.utwente.nl/user/s2847868/S1S2_meansTest"
labels_path = "hdfs://ctit048.ewi.utwente.nl/user/s2847868/phase_diff"
spark = SparkSession(sc)

df_pressure = spark.read.csv(pressure_path, header=True, inferSchema=True)
df_labels = spark.read.csv(labels_path, inferSchema=True).select(col("_c0").alias("index"),col("_c1").alias("label"))
df_combined = df_pressure.join(df_labels, df_pressure["index"] == df_labels["index"])

#transform p1 and p2 into ml vectors for training
va = VectorAssembler(inputCols=['p1', 'p2'], outputCol="features")
df2 = va.transform(df_combined)

#create model
dt = DecisionTreeRegressor(maxDepth=4, varianceCol="variance")

train, test = df2.select(col("features"), col("label")).randomSplit([0.8,0.2], seed)
model = dt.fit(train)

evaluator = RegressionEvaluator()
print("Depth 4")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))

dt = DecisionTreeRegressor(maxDepth=3, varianceCol="variance")

train, test = df2.select(col("features"), col("label")).randomSplit([0.8,0.2], seed)
model = dt.fit(train)

evaluator = RegressionEvaluator()
print("Depth 3")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))


glr = GeneralizedLinearRegression(family="gaussian", link="identity", linkPredictionCol="p")
model = glr.fit(train)
evaluator = RegressionEvaluator()

print("Gaussian linear regression")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))