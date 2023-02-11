from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression


from pyspark.ml.linalg import Vectors

features_path = "hdfs://ctit048.ewi.utwente.nl/user/s2640996/phase_diff"

spark = SparkSession.builder.appName('project').getOrCreate()
df = spark.read.csv(features_path)


lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal", weightCol="weight")
model = lr.fit(df)


df.show()   