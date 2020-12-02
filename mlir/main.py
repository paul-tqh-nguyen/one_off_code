
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

def example_1():    
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(units=1)
    ])
    
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    x = np.arange(2, dtype=float).reshape([1,2])
    
    print(f"model.predict(x) {repr(model.predict(x))}")
    print(f"model.summary() {repr(model.summary())}")


if __name__ == '__main__':
    print('\n'*80)
    
    g = tf.Graph()
    with g.as_default():
        inputs = tf.keras.Input(shape=(2,)) 
        x = tf.keras.layers.Dense(units=1)(inputs)
    graph_def = g.as_graph_def()
    mlir_text = tf.mlir.experimental.convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline')
    print(f"mlir_text \n{mlir_text}")
