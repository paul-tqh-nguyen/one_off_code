'#!/usr/bin/python3 -OO' # @todo use this

'''
'''

# @todo update doc string

###########
# Imports #
###########

import pytorch_lightning as pl

from misc_utilities import *
from global_values import *
from models import LinearColaborativeFilteringModel
from data_modules import AnimeRatingDataModule

############
# Training #
############

class PrintingCallback(pl.Callback):

    def __init__(self, checkpoint_callback: pl.callbacks.ModelCheckpoint):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
    
    def on_init_start(self, trainer: pl.Trainer) -> None:
        LOGGER.info('')
        LOGGER.info('Initializing trainer.')
        LOGGER.info('')
        return
    
    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Model: ')
        LOGGER.info(model)
        LOGGER.info('')
        LOGGER.info(f'Training GPUs: {trainer.gpus}')
        LOGGER.info(f'Number of Epochs: {model.hparams.number_of_epochs:,}')
        LOGGER.info(f'Learning Rate: {model.hparams.learning_rate:,}')
        LOGGER.info(f'Number of Animes: {model.hparams.number_of_animes:,}')
        LOGGER.info(f'Number of Users: {model.hparams.number_of_users:,}')
        LOGGER.info(f'Embedding Size: {model.hparams.embedding_size:,}')
        LOGGER.info(f'Regularization Factor: {model.hparams.regularization_factor:,}')
        LOGGER.info(f'Dropout Probability: {model.hparams.dropout_probability:,}')
        LOGGER.info('')
        LOGGER.info('Data:')
        LOGGER.info('')
        LOGGER.info(f'Training Batch Size: {trainer.train_dataloader.batch_size:,}')
        LOGGER.info(f'Validation Batch Size: {only_one(trainer.val_dataloaders).batch_size:,}')
        LOGGER.info('')
        LOGGER.info(f'Training Batch Count: {len(trainer.train_dataloader):,}')
        LOGGER.info(f'Validation Batch Count: {len(only_one(trainer.val_dataloaders)):,}')
        LOGGER.info('')
        LOGGER.info(f'Training Example Count: {len(trainer.train_dataloader)*trainer.train_dataloader.batch_size:,}')
        LOGGER.info(f'Validation Example Count: {len(only_one(trainer.val_dataloaders))*only_one(trainer.val_dataloaders).batch_size:,}')
        LOGGER.info('')
        LOGGER.info('Starting training.')
        LOGGER.info('')
        return
    
    def on_train_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Training complete.')
        LOGGER.info('')
        return

    def on_test_start(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Starting testing.')
        LOGGER.info('')
        LOGGER.info(f'Testing Batch Size: {only_one(trainer.test_dataloaders).batch_size:,}')
        LOGGER.info(f'Testing Example Count: {len(only_one(trainer.test_dataloaders))*only_one(trainer.test_dataloaders).batch_size:,}')
        LOGGER.info(f'Testing Batch Count: {len(only_one(trainer.test_dataloaders)):,}')
        LOGGER.info('')
        return
    
    def on_test_end(self, trainer: pl.Trainer, model: pl.LightningDataModule) -> None:
        LOGGER.info('')
        LOGGER.info('Testing complete.')
        LOGGER.info('')
        LOGGER.info(f'Best validation model checkpoint saved to {self.checkpoint_callback.best_model_path} .')
        LOGGER.info('')
        return

def checkpoint_directory_from_hyperparameters(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, regularization_factor: float, dropout_probability: float) -> str:
    checkpoint_dir = f'./checkpoints/' \
        f'linear_cf_lr_{learning_rate:.5g}_' \
        f'epochs_{number_of_epochs}_' \
        f'batch_{batch_size}_' \
        f'gradient_clip_{gradient_clip_val:.3g}_' \
        f'embed_{embedding_size}_' \
        f'regularization_{regularization_factor:.5g}_' \
        f'dropout_{dropout_probability:.5g}'
    return checkpoint_dir

def train_model(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, regularization_factor: float, dropout_probability: float, gpus: List[int]) -> float:

    checkpoint_dir = checkpoint_directory_from_hyperparameters(learning_rate, number_of_epochs, batch_size, gradient_clip_val, embedding_size, regularization_factor, dropout_probability)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:03d}_{val_checkpoint_on}'),
            save_top_k=1,
            verbose=True,
            save_last=True,
            monitor='val_checkpoint_on',
            mode='min',
        )

    trainer = pl.Trainer(
        callbacks=[PrintingCallback(checkpoint_callback)],
        max_epochs=number_of_epochs,
        min_epochs=number_of_epochs,
        gradient_clip_val=gradient_clip_val,
        terminate_on_nan=True,
        gpus=gpus,
        distributed_backend='dp',
        deterministic=False,
        # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
        logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='linear_cf_model'),
        checkpoint_callback=checkpoint_callback,
    )

    data_module = AnimeRatingDataModule(batch_size, NUM_WORKERS)
    data_module.prepare_data()
    data_module.setup()

    model = LinearColaborativeFilteringModel(
        learning_rate = learning_rate,
        number_of_epochs = number_of_epochs,
        number_of_training_batches = len(data_module.training_dataloader),
        number_of_animes = data_module.number_of_animes,
        number_of_users = data_module.number_of_users,
        embedding_size = embedding_size,
        regularization_factor = regularization_factor,
        dropout_probability = dropout_probability,
    )

    trainer.fit(model, data_module)
    test_results = only_one(trainer.test(model, datamodule=data_module, verbose=False, ckpt_path=checkpoint_callback.best_model_path))
    best_validation_loss = checkpoint_callback.best_model_score.item()
    
    output_json_file_location = os.path.join(checkpoint_dir, 'result_summary.json')
    with open(output_json_file_location, 'w') as file_handle:
        json.dump({
            'testing_loss': test_results['testing_loss'],
            'testing_regularization_loss': test_results['testing_regularization_loss'],
            'testing_mse_loss': test_results['testing_mse_loss'],
            
            'best_validation_loss': best_validation_loss,
            'best_validation_model_path': checkpoint_callback.best_model_path,
            
            'learning_rate': learning_rate,
            'number_of_epochs': number_of_epochs,
            'batch_size': batch_size,
            'number_of_animes': data_module.number_of_animes,
            'number_of_users': data_module.number_of_users,
            'embedding_size': embedding_size,
            'regularization_factor': regularization_factor,
            'dropout_probability': dropout_probability,

            'training_set_batch_size': data_module.training_dataloader.batch_size,
            'training_set_batch_count': len(data_module.training_dataloader),
            'validation_set_batch_size': data_module.validation_dataloader.batch_size,
            'validation_set_batch_count': len(data_module.validation_dataloader),
            'testing_set_batch_size': data_module.test_dataloader().batch_size,
            'testing_set_batch_count': len(data_module.test_dataloader()),
        }, file_handle, indent=4)

    return best_validation_loss
