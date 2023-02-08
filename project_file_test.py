from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,explode, monotonically_increasing_id
from pyspark.sql.types import IntegerType, DoubleType, StructField, StructType, ArrayType
from pyspark.ml.linalg import Vectors

import os
import math

base_path = "hdfs://ctit048.ewi.utwente.nl/user/s2640996"

sensor_name = "Sensor3"
labels_name = "FlowLabel.csv"

#files = os.listdir('user//s2640996/Sensor3')
sc = SparkContext(appName='feature_extraction')

spark = SparkSession.builder.appName('project').getOrCreate()


empty_rdd = sc.emptyRDD()
columns = StructType([StructField('ID', IntegerType(), False),
		      StructField('PhaseDelta', DoubleType(), False)])

feature_df = spark.createDataFrame(data=empty_rdd, schema=columns)
feature_df.show()
counter = 0


### READ LABEL
labels = spark.read.csv(os.path.join(base_path, labels_name), header=True).collect()


for measurement in labels:
     if counter > 1:
          break

     c1 = spark.read.text(os.path.join(base_path,'Sensor3/row')+str(measurement['Index'])+'.csv', lineSep=',').toDF('C1')
     c2 = spark.read.text(os.path.join(base_path,'Sensor4/row')+str(measurement['Index'])+'.csv', lineSep=',').toDF('C2')
     
     c1 = c1.withColumn('C1', c1.C1.cast('double'))
     c1 = c1.select('*').withColumn('id', monotonically_increasing_id())

     c2 = c2.withColumn('C2', c2.C2.cast('double'))

     c2 = c2.select('*').withColumn('id', monotonically_increasing_id())

     c1c2 = c1.join(c2, on=['id'])
     print("c1c2: ",c1c2.count()," ",len(c1c2.columns))
     print("covariance: ",c1c2.cov('C1', 'C2'))
     feature = math.acos(c1c2.cov('C1', 'C2'))
     temp_df = spark.createDataFrame([(measurement['Index'], feature)], ['ID', 'PhaseDelta'])
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
