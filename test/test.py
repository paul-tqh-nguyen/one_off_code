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
    words = sc.parallelize (
        ["scala", 
         "java", 
         "hadoop", 
         "spark", 
         "akka",
         "spark vs hadoop", 
         "pyspark",
         "pyspark and spark"]
    )
    result = words.map(lambda x: x+'123').collect()
    print(f"result {repr(result)}")
    return
            
if __name__ == '__main__':
    main()
