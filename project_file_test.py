from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,explode, monotonically_increasing_id
from pyspark.sql.types import *
import math
import numpy as np
import os

#files = os.listdir('user//s2640996/Sensor3')
spark = SparkContext(appName='feature_extraction')

sc = SparkSession.builder.appName('project').getOrCreate()

empty_rdd =spark.emptyRDD()
columns = StructType([StructField('ID', IntegerType(), False),
		      StructField('PhaseDelta', DoubleType(), False)])

feature_df = sc.createDataFrame(data=empty_rdd, schema=columns)
feature_df.show()
counter = 0

### READ LABEL
labels = sc.read.csv('FlowLabel.csv', header=True).collect()


for measurement in labels:
     if counter > 600:
          break
     c1 = sc.read.text('Sensor3/row'+str(measurement['Index'])+'.csv', lineSep=',').toDF('C1')
     c2 = sc.read.text('Sensor4/row'+str(measurement['Index'])+'.csv', lineSep=',').toDF('C2')

     c1 = c1.withColumn('C1', c1.C1.cast('double'))
     c1 = c1.select('*').withColumn('id', monotonically_increasing_id())

     c2 = c2.withColumn('C2', c2.C2.cast('double'))

     c2 = c2.select('*').withColumn('id', monotonically_increasing_id())

     c1c2 = c1.join(c2, on=['id'])
     feature = math.acos(c1c2.cov('C1', 'C2'))
     temp_df = sc.createDataFrame([(measurement['Index'], feature)], ['ID', 'PhaseDelta'])
     feature_df = feature_df.union(temp_df)
     counter = counter + 1

feature_df.show()
#c1 = spark.read.text('Sensor3/row4433.csv', lineSep=',').toDF('C1')
#c2 = spark.read.text('Sensor4/row4433.csv', lineSep=',').toDF('C2')

#c1 = c1.withColumn('C1', c1.C1.cast('double'))
#c1 = c1.select('*').withColumn('id', monotonically_increasing_id())

#c2 = c2.withColumn('C2', c2.C2.cast('double'))

#c2 = c2.select('*').withColumn('id', monotonically_increasing_id())

#c1c2 = c1.join(c2, on=['id'])

#print(math.acos(c1c2.cov('C1', 'C2')))
