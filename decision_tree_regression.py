from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
seed = 284

sc = SparkContext(appName='feature_extraction')

pressure_path = "hdfs://ctit048.ewi.utwente.nl/user/s2847868/S1S2_meansTest"
phase_diff_path = "hdfs://ctit048.ewi.utwente.nl/user/s2847868/phase_diff"
flowlabels_path = "hdfs://ctit048.ewi.utwente.nl/user/s2640996/FlowLabel.csv"
magnitude_path = "hdfs://ctit048.ewi.utwente.nl/user/s2640996/magn_test12"
spark = SparkSession(sc)

df_flow= spark.read.csv(flowlabels_path, header=True, inferSchema=True)
df_magnitude = spark.read.csv(magnitude_path, inferSchema=True).select(col("_c0").alias("Index"),col("_c1").alias("C1"),col("_c3").alias("C2"))
df_pressure = spark.read.csv(pressure_path, header=True, inferSchema=True) ## with non capital i in index
df_phase = spark.read.csv(phase_diff_path, inferSchema=True).select(col("_c0").alias("Index"),col("_c1").alias("phase_difference"))

df_combined = df_flow.join(df_magnitude, df_flow["Index"] == df_magnitude["Index"])\
    .join(df_pressure, df_flow["Index"] == df_pressure["index"])\
    .join(df_phase, df_flow["Index"] == df_phase["Index"])

#transform p1 and p2 into ml vectors for training


va = VectorAssembler(inputCols=['p1', 'p2', 'C1', 'C2', 'phase_difference'], outputCol="params")
df_combined = va.transform(df_combined)
standardScaler = StandardScaler(inputCol='params', outputCol="features")
model = standardScaler.fit(df_combined)
df2 = model.transform(df_combined)
# df2 = va.transform(df_combined)


#create model
dt = DecisionTreeRegressor(maxDepth=5, varianceCol="variance")

train, test = df2.select(col("features"), col("Label").alias('label')).randomSplit([0.8,0.2], seed)
model = dt.fit(train)

evaluator = RegressionEvaluator()
print("Depth 5")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))
print("Features importances ", model.featureImportances)
prediction_results = "Decisiontree_regression_predictions_5"
model.transform(test).write.parquet(prediction_results)
#depth 4

dt = DecisionTreeRegressor(maxDepth=4, varianceCol="variance")

model = dt.fit(train)

evaluator = RegressionEvaluator()
print("Depth 4")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))
print("Features importances ", model.featureImportances)
prediction_results = "Decisiontree_regression_predictions_4"
model.transform(test).write.parquet(prediction_results)
#depth 3

dt = DecisionTreeRegressor(maxDepth=3, varianceCol="variance")

model = dt.fit(train)

evaluator = RegressionEvaluator()
print("Depth 3")
evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))
print("Features importances ", model.featureImportances)
prediction_results = "Decisiontree_regression_predictions_3"
model.transform(test).write.parquet(prediction_results)


print("Gaussian linear regression")
glr = GeneralizedLinearRegression(family="gaussian", link="identity", linkPredictionCol="p")
model = glr.fit(train)
evaluator = RegressionEvaluator()

evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))
prediction_results = "Gaussian_linear_predictions"
model.transform(test).write.parquet(prediction_results)

print("Ridge Regression")
##Ridge regression requires lambda > 0  and alpha = 0  otherwise we have lasso regression
lambd = 0.01
alpha = 0

lr = LinearRegression(maxIter=10, regParam=lambd, elasticNetParam=alpha)
model = lr.fit(train)
evaluator = RegressionEvaluator()


evaluator.evaluate(model.transform(test))
print("Mean absolute error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mae"}))
print("Root mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "rmse"}))
print("Mean squared error: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "mse"}))
print("R2 score: ", evaluator.evaluate(model.transform(test), {evaluator.metricName: "r2"}))
prediction_results = "Ridge_Regression"
model.transform(test).write.parquet(prediction_results)

