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

    aaa = sc.parallelize([('a', 1), ('b', 2)])
    bbb = sc.parallelize([('b', 3), ('d', 4)])
    result = aaa.join(bbb).collect()
    print(f"result {repr(result)}")
    
    return
            
if __name__ == '__main__':
    print('\n' * 100)
    main()
