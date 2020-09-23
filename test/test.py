#!/usr/bin/python3 -O

###########
# Imports #
###########

import os
import pyspark

###########
# Globals #
###########

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

##########
# Driver #
##########

def main() -> None:
    logFile = f'file://{CURRENT_DIRECTORY}/test.py'
    conf = pyspark.SparkConf().set('spark.driver.host','127.0.0.1')
    sc = pyspark.SparkContext(master='local', appName='first app', conf=conf)
    logData = sc.textFile(logFile).cache()
    numAs = logData.filter(lambda s: 'a' in s).count()
    numBs = logData.filter(lambda s: 'b' in s).count()
    print(f'Lines with a: {numAs}, lines with b: {numBs}')
    return
            
if __name__ == '__main__':
    main()
