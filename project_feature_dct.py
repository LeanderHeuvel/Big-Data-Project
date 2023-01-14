from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler, DCT

from pyspark.ml.linalg import Vectors
# https://stackoverflow.com/questions/32788387/pipelinedrdd-object-has-no-attribute-todf-in-pyspark
spark = SparkContext(appName='feature_extraction')

sc = SparkSession(spark)

C1 = spark.wholeTextFiles('Sensor3')

df1 = C1.map(lambda (path, data) : (path.split('row')[1], Vectors.dense(data.split(',')))).toDF()

dct = DCT(inverse=False, inputCol='_2', outputCol='result')

df2 = dct.transform(df1)

print(df2.select('result').head(1))
