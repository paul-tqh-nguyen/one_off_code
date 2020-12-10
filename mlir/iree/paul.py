
#####################
# Imports & Globals #
#####################

import os
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.image

import pyiree.rt
import pyiree.compiler2
import pyiree.compiler2.tf

#####################
# TensorFlow Module #
#####################

class EdgeDetectionModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([1, 128, 128, 1], tf.float32)])
  def edge_detect_sobel_operator(self, image):
    # https://en.wikipedia.org/wiki/Sobel_operator
    sobel_x = tf.constant([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]],
                          dtype=tf.float32, shape=[3, 3, 1, 1])    
    sobel_y = tf.constant([[ 1.0,  2.0,  1.0],
                           [ 0.0,  0.0,  0.0],
                           [-1.0, -2.0, -1.0]],
                          dtype=tf.float32, shape=[3, 3, 1, 1])
    gx = tf.nn.conv2d(image, sobel_x, 1, "SAME")
    gy = tf.nn.conv2d(image, sobel_y, 1, "SAME")
    return tf.math.sqrt(gx * gx + gy * gy)

def load_input_image():
    path_to_image = tf.keras.utils.get_file(
        'YellowLabradorLooking_new.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    )
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (128, 128))
    image = image[tf.newaxis, :]
    image = image.numpy()
    return image

##########
# Driver #
##########

if __name__ == '__main__':
    
    assert '/home/pnguyen/miniconda3/envs/mlir/lib' in os.environ['LD_LIBRARY_PATH']

    with tempfile.TemporaryDirectory() as temporary_directory:
        
        module = EdgeDetectionModule()

        saved_module_path = os.path.join(temporary_directory, 'saved_module.sm')
        
        tf.saved_model.save(
            module,
            saved_module_path,
            options=tf.saved_model.SaveOptions(save_debug_info=True)
        )
        
        mlir_text = pyiree.compiler2.tf.compile_saved_model(saved_module_path, import_only=True).decode('utf-8')

        target_backends = pyiree.compiler2.tf.DEFAULT_TESTING_BACKENDS
        binary_text: bytes = pyiree.compiler2.tf.compile_module(
            module,
            target_backends=target_backends
        )

        vm_module = pyiree.rt.VmModule.from_flatbuffer(binary_text)

        config = pyiree.rt.Config()

        ctx = pyiree.rt.SystemContext(config=config)
        ctx.add_module(vm_module)

        module_func = ctx.modules.module['edge_detect_sobel_operator']

        input_image = load_input_image()
        
        result = module_func(input_image)

        matplotlib.image.imsave('input_image.png', input_image.reshape([128, 128]))
        matplotlib.image.imsave('result.png', result.reshape([128, 128]))
        
        print('\n'*10)
        print()
        print(f'mlir_text: \n\n'+mlir_text)
        print()
        print(f'target_backends {target_backends}')
        print()
        print(f'binary_text.hex() {repr(binary_text.hex())}')
        print()
        print(f'vm_module {repr(vm_module)}')
        print()
        print(f'config {repr(config)}')
        print()
        print(f'ctx {repr(ctx)}')
        print()
        print(f'module_func {repr(module_func)}')
        print()
        print(f'input_image.shape {repr(input_image.shape)}')
        print()
        print(f"type(result) {repr(type(result))}")
        print(f'result.shape {repr(result.shape)}')
        print()
    
    # breakpoint()
