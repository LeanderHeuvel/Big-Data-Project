from pyspark import SparkContext
from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler, DCT
import numpy as np
from pyspark.ml.stat import Summarizer

from pyspark.ml.linalg import Vectors, VectorUDT
# https://stackoverflow.com/questions/32788387/pipelinedrdd-object-has-no-attribute-todf-in-pyspark
sc = SparkContext(appName='feature_extraction')

spark = SparkSession(sc)

base_path = "/user/s2640996/"

C1 = sc.wholeTextFiles(base_path+'Sensor1', minPartitions=10)
C2 = sc.wholeTextFiles(base_path+'Sensor2', minPartitions=10)

rdd1 = C1.map(lambda (path, data): (path.split('row')[1][:-4], float(np.mean(list(map(float, data.split(',')))))))
rdd2 = C2.map(lambda (path, data): (path.split('row')[1][:-4], float(np.mean(list(map(float, data.split(',')))))))

rdd1 = rdd1.repartition(numPartitions=20)
rdd2 = rdd2.repartition(numPartitions=20)

rdd3 = rdd1.join(rdd2)

meansDFStruct = rdd3.toDF()

meansDF =  meansDFStruct.select(col('_1').alias('index'), col('_2')['_1'].alias('p1'), col('_2')['_2'].alias('p2'))

meansDF.write.csv('S1S2_meansTest', header=True)
