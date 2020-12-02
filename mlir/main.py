
'''
'''

# @todo fill in doc string

###########
# Imports #
###########

import numpy as np
import tensorflow as tf

# @todo verify that these imports are used

##########
# Driver #
##########

if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        inputs = tf.keras.Input(shape=(2,)) 
        x = tf.keras.layers.Dense(units=1)(inputs)
    graph_def = g.as_graph_def()
    mlir_text = tf.mlir.experimental.convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline')
    print('\n'*80)
    print(f"mlir_text {mlir_text}")
