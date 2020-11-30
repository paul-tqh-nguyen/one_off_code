
'''
'''

# @todo fill in doc string

###########
# Imports #
###########

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# @todo verify that these imports are used

##########
# Driver #
##########

if __name__ == '__main__':
    
    model = layers.Dense(units=1)
    
    model.summary()
