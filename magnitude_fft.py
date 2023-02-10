from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler, DCT
from pyspark.sql.types import FloatType, DoubleType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
# https://stackoverflow.com/questions/32788387/pipelinedrdd-object-has-no-attribute-todf-in-pyspark
spark = SparkContext(appName='feature_extraction').getOrCreate()

sc = SparkSession(spark)

base_path = "/user/s2640996/"

sensor_name = "Sensor3"

C1 = spark.wholeTextFiles(base_path+'Sensor3', minPartitions=300)
C2 = spark.wholeTextFiles(base_path+'Sensor4', minPartitions=300)

# returns a row with filename (_1) and a densevector(_2) containing all measurements

## current runtime

df1 = C1.map(lambda (path, data) : (path.split('row')[1][:-4], data.split(','))).toDF().repartition(300, col('_1'))
df2 = C2.map(lambda (path, data) : (path.split('row')[1][:-4], data.split(','))).toDF().repartition(300, col('_1'))

df1 = df1.select(col('_1'), col('_2').cast(ArrayType(DoubleType())))
df2 = df2.select(col('_1'), col('_2').cast(ArrayType(DoubleType())))

def magnitude(data):
	return float(max(abs(np.fft.fft(data))))

magn_udf = udf(magnitude)

# df1_magn = df1.map(lambda (index, data): (index, max(abs(np.fft.fft(data)))))

df_dct1 = df1.withColumn('valuesC1', magn_udf(df1['_2'])).select(col('_1').alias('C1I'), col('valuesC1'))
df_dct2 = df2.withColumn('valuesC2', magn_udf(df2['_2'])).select(col('_1').alias('C2I'), col('valuesC2'))

combined = df_dct1.join(df_dct2, df_dct1.C1I == df_dct2.C2I)
combined.write.csv('magn_test12')
# print(df2.select('result').head(1))


