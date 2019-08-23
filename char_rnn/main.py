#!/usr/bin/python3

import numpy as np
import os
import time
import tensorflow as tf
tf.enable_eager_execution()

def gather_entire_text():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    return text

def generate_character_to_index_mappings(text):
    vocabulary = sorted(set(text))
    character_to_integer_map = {u:i for i, u in enumerate(vocabulary)}
    integer_to_character_map = np.array(vocabulary)
    return character_to_integer_map, integer_to_character_map, vocabulary

def convert_text_to_integer_representation(text, character_to_integer_map):
    integer_list_representation = [character_to_integer_map[character] for character in text]
    integer_numpy_array_representation = np.array(integer_list_representation)
    return integer_numpy_array_representation

SEQUENCE_LENGTH = 100

def main():
    
    text = gather_entire_text()
    character_to_integer_map, integer_to_character_map, vocabulary = generate_character_to_index_mappings(text)
    text_as_int = convert_text_to_integer_representation(text, character_to_integer_map)
    num_iterations_per_epoch = len(text)//SEQUENCE_LENGTH
    
    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    print("Here are some of the first characters in the dataset.")
    for i in char_dataset.take(5):
        print(integer_to_character_map[i.numpy()])
    
    sequences = char_dataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True)
    
    print("Here are some of the sequences.")
    for item in sequences.take(5):
        print(repr(''.join(integer_to_character_map[item.numpy()])))
    
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
    dataset = sequences.map(split_input_target)
    
    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', repr(''.join(integer_to_character_map[input_example.numpy()])))
        print ('Target data:', repr(''.join(integer_to_character_map[target_example.numpy()])))
    
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(integer_to_character_map[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(integer_to_character_map[target_idx])))
    
    # Batch size
    BATCH_SIZE = 64
    steps_per_epoch = num_iterations_per_epoch//BATCH_SIZE
    
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    print("dataset")
    print(dataset)
    
    # Length of the vocabulary in chars
    vocabulary_size = len(vocabulary)
    
    # The embedding dimension
    embedding_dim = 256
    
    # Number of RNN units
    rnn_units = 1024
    
    if tf.test.is_gpu_available():
        print("We are training on the GPU!")
        rnn = tf.keras.layers.CuDNNGRU
    else:
        print("We can't train via GPU!")
        import functools
        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')
    
    def build_model(vocabulary_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocabulary_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            rnn(rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True),
            tf.keras.layers.Dense(vocabulary_size)
        ])
        return model
    
    model = build_model(
        vocabulary_size = vocabulary_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)
    
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocabulary_size)")
    
    model.summary()
    
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
    
    print("sampled_indices")
    print(sampled_indices)
    
    print("Input: \n", repr("".join(integer_to_character_map[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(integer_to_character_map[sampled_indices ])))
    
    def loss(labels, logits):
      return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
    example_batch_loss  = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocabulary_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    
    model.compile(
        optimizer = tf.train.AdamOptimizer(),
        loss = loss)
    
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    EPOCHS=3
    
    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
    
    print("tf.train.latest_checkpoint(checkpoint_dir)")
    print(tf.train.latest_checkpoint(checkpoint_dir))
    
    model = build_model(vocabulary_size, embedding_dim, rnn_units, batch_size=1)
    
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    model.build(tf.TensorShape([1, None]))
    
    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)
        
        # Number of characters to generate
        num_generate = 1000
        
        # Converting our start string to numbers (vectorizing)
        input_eval = [character_to_integer_map[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        # Empty string to store our results
        text_generated = []
        
        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0
        
        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            
            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
            
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(integer_to_character_map[predicted_id])
            
        return (start_string + ''.join(text_generated))
    
    print(generate_text(model, start_string=u"ROMEO: "))     

if __name__ == '__main__':
    main()
