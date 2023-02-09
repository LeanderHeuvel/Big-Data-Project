from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler, DCT

from pyspark.ml.linalg import Vectors, VectorUDT
# https://stackoverflow.com/questions/32788387/pipelinedrdd-object-has-no-attribute-todf-in-pyspark
spark = SparkContext(appName='feature_extraction')

sc = SparkSession(spark)

base_path = "/user/s2640996/"

sensor_name = "Sensor3"

C1 = spark.wholeTextFiles(base_path+'Sensor3', minPartitions=200)
C2 = spark.wholeTextFiles(base_path+'Sensor4', minPartitions=200)

# returns a row with filename (_1) and a densevector(_2) containing all measurements

## current runtime

df1 = C1.map(lambda (path, data) : (path.split('row')[1][:-4], Vectors.dense(data.split(',')))).toDF().repartition(200, col('_1'))
df2 = C2.map(lambda (path, data) : (path.split('row')[1][:-4], Vectors.dense(data.split(',')))).toDF().repartition(200, col('_1'))

dct = DCT(inverse=False, inputCol='_2', outputCol='result')

df_dct1 = dct.transform(df1)
df_dct2 = dct.transform(df2)

def max_val(vector):
	return  max(vector)

max_udf = udf(max_val)

df_dct1 = df_dct1.withColumn('valuesC1', max_udf(df_dct1['result']))
df_dct2 = df_dct2.withColumn('valuesC2', max_udf(df_dct2['result']))

combined = df_dct1.join(df_dct2, df_dct1._1 == df_dct2._1)
combined.select(col('_1'), col('valuesC1'), col('valuesC2')).write.csv('magn_test2')
# print(df2.select('result').head(1))


