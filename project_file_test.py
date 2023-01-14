from pyspark.sql import SparkSession
from pyspark.sql.functions import col,explode, monotonically_increasing_id
import math

spark = SparkSession.builder.appName('project').getOrCreate()

c1 = spark.read.text('Sensor3/row4433.csv', lineSep=',').toDF('C1')
c2 = spark.read.text('Sensor4/row4433.csv', lineSep=',').toDF('C2')

c1 = c1.withColumn('C1', c1.C1.cast('double'))
c1 = c1.select('*').withColumn('id', monotonically_increasing_id())

c2 = c2.withColumn('C2', c2.C2.cast('double'))

c2 = c2.select('*').withColumn('id', monotonically_increasing_id())

c1c2 = c1.join(c2, on=['id'])

print(math.acos(c1c2.cov('C1', 'C2')))
