#!/usr/bin/python3 -O

###########
# Imports #
###########

import os
import pyspark

##########
# Driver #
##########

def init_pyspark() -> None:
    # os.environ[''] = ''
    # export SPARK_HOME = /home/hadoop/spark-2.1.0-bin-hadoop2.7
    # export PATH = $PATH:/home/hadoop/spark-2.1.0-bin-hadoop2.7/bin
    # export PYTHONPATH = $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH
    # export PATH = $SPARK_HOME/python:$PATH
    return

def main() -> None:
    init_pyspark()
    logFile = 'file:///Users/pnguyen/code/one_off_code/test/test.py'
    conf = pyspark.SparkConf().set('spark.driver.host','127.0.0.1')
    sc = pyspark.SparkContext(master='local', appName='first app', conf=conf)
    logData = sc.textFile(logFile).cache()
    numAs = logData.filter(lambda s: 'a' in s).count()
    numBs = logData.filter(lambda s: 'b' in s).count()
    print(f'Lines with a: {numAs}, lines with b: {numBs}')
    return
            
if __name__ == '__main__':
    main()
