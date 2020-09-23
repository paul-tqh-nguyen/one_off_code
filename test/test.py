#!/usr/bin/python3 -O

###########
# Imports #
###########

from pyspark import SparkContext

##########
# Driver #
##########

def main():
    logFile = "file:///home/hadoop/spark-2.1.0-bin-hadoop2.7/README.md"  
    sc = SparkContext("local", "first app")
    logData = sc.textFile(logFile).cache()
    numAs = logData.filter(lambda s: 'a' in s).count()
    numBs = logData.filter(lambda s: 'b' in s).count()
    print("Lines with a: {numAs}, lines with b: {numBs}")
    return
            
if __name__ == '__main__':
    main()
