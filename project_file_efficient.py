from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,explode, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, StructField, StructType, ArrayType
from pyspark.ml.linalg import Vectors

import math
import numpy as np
import os

base_path = "hdfs://ctit048.ewi.utwente.nl/user/s2640996"

sensor_name = "Sensor3"
labels_name = "FlowLabel.csv"

sc = SparkContext(appName='feature_extraction')

spark = SparkSession.builder.appName('project').getOrCreate()


sensor3 = sc.wholeTextFiles(os.path.join(base_path,"Sensor3"))
sensor3 = sensor3.map(lambda (path, data) : (path.split('row')[1], data.split(','))).toDF()
sensor3 = sensor3.select(col('_1'), col("_2").alias("sensor3_measurement").cast(ArrayType(DoubleType())))

sensor4 = sc.wholeTextFiles(os.path.join(base_path,"Sensor4"))
sensor4 = sensor4.map(lambda (path, data) : (path.split('row')[1], data.split(','))).toDF()
sensor4 = sensor4.select(col('_1'), col("_2").alias("sensor4_measurement").cast(ArrayType(DoubleType())))

sensors = sensor4.join(sensor3, on=['_1']).rdd
cov = sensors.map(lambda x: (x[0], np.cov(x[1], x[2]).tolist()))
acos = cov.map(lambda x: math.acos(x[0],x[1]))
print(acos.count())