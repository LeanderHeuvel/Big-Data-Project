from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,explode, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, StructField, StructType, ArrayType
from pyspark.ml.linalg import Vectors

import math
import numpy as np
#import os
#from os import path

base_path = "/user/s2640996/"

sensor_name = "Sensor3"
labels_name = "FlowLabel.csv"

sc = SparkContext(appName='feature_extraction')

spark = SparkSession.builder.appName('project').getOrCreate()


sensor3 = sc.wholeTextFiles(base_path+"Sensor3", minPartitions=300)
sensor3 = sensor3.map(lambda (path, data) : (path.split('row')[1][:-4], data.split(','))).toDF().repartition(300,  col('_1'))
sensor3 = sensor3.select(col('_1'), col("_2").alias("sensor3_measurement").cast(ArrayType(DoubleType())))
#sensor3 = sensor3.limit(10)


sensor4 = sc.wholeTextFiles(base_path+"Sensor4", minPartitions=300)
sensor4 = sensor4.map(lambda (path, data) : (path.split('row')[1][:-4], data.split(','))).toDF().repartition(300, col('_1'))
sensor4 = sensor4.select(col('_1'), col("_2").alias("sensor4_measurement").cast(ArrayType(DoubleType())))
#sensor4 = sensor4.limit(10)

sensors = sensor4.join(sensor3, on=['_1']).rdd

#sensors.show()
# print(len(sensors.take(1)[0][1]))
cov = sensors.map(lambda x: (x[0], np.corrcoef(x[1], x[2])[0][1]))
#print(cov)
#print("cov first take",cov.take(1))

acos = cov.map(lambda x: (x[0], math.acos(x[1]))).toDF()

acos.write.csv('phase_diff')

#print(acos)
#print(acos.take(1))

