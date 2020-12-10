
'''
'''

# @todo fill in doc string

###########
# Imports #
###########

import tempfile
import subprocess

import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# @todo verify that these imports are used

#############
# Utilities #
#############

def execute_shell_command(shell_command: str) -> str:
    process = subprocess.Popen(shell_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_string, _ = process.communicate()
    stdout_string = stdout_string.decode("utf-8")
    return stdout_string

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
    with tempfile.NamedTemporaryFile(suffix='bef') as file_handle:
        file_handle.write(mlir_text.encode())
        file_handle.seek(0)
        execution_output = execute_shell_command(f'./runtime/bazel-bin/tools/bef_executor {file_handle.name}')
        print(f"mlir_text {mlir_text}\n")
        print(f"execution_output: \n\n{execution_output}")
