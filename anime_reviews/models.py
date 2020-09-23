#!/usr/bin/python3 -OO

'''

This module contains collaborative filtering models to learn representations of users and animes.

Sections:
* Imports
* Globals
* Abstract Collaborative Filtering Model
* Linear Model
* Deep Concatenation Model

'''

###########
# Imports #
###########

import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple
from typing_extensions import Literal

import torch
from torch import nn
import transformers
import pytorch_lightning as pl

from misc_utilities import *
from global_values import *
from data_modules import AnimeRatingDataModule

###########
# Globals #
###########

MSE_LOSS = nn.MSELoss(reduction='none')

##########################################
# Abstract Collaborative Filtering Model #
##########################################

class AbstractColaborativeFilteringModel(pl.LightningModule, ABC):
        
    @abstractmethod
    def forward(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def backward(self, _trainer: pl.Trainer, loss: torch.Tensor, _optimizer: torch.optim.Optimizer, _optimizer_idx: int) -> None:
        del _trainer, _optimizer, _optimizer_idx
        loss.mean().backward()
        return

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer: torch.optim.Optimizer = transformers.AdamW(self.parameters(), lr=self.hparams.learning_rate, correct_bias=False)
        scheduler: torch.optim.lr_scheduler.LambdaLR = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.hparams.number_of_training_batches*self.hparams.number_of_epochs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
    def _get_batch_loss(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_scores = batch_dict['rating']
        batch_size = only_one(target_scores.shape)
        predicted_scores, regularization_loss = self(batch_dict)
        assert tuple(predicted_scores.shape) == (batch_size,)
        assert tuple(regularization_loss.shape) == (batch_size,)
        mse_loss = MSE_LOSS(predicted_scores, target_scores)
        assert tuple(mse_loss.shape) == (batch_size,)
        assert mse_loss.ge(0).all()
        loss = regularization_loss + mse_loss
        assert tuple(loss.shape) == (batch_size,)
        assert tuple(loss.shape) == tuple(mse_loss.shape) == tuple(regularization_loss.shape) == (batch_size,)
        return loss, mse_loss, regularization_loss
    
    def training_step(self, batch_dict: dict, _: int) -> pl.TrainResult:
        loss, _, _ = self._get_batch_loss(batch_dict)
        result = pl.TrainResult(minimize=loss)
        return result

    def training_step_end(self, training_step_result: pl.TrainResult) -> pl.TrainResult:
        assert len(training_step_result.minimize.shape) == 1
        mean_loss = training_step_result.minimize.mean()
        result = pl.TrainResult(minimize=mean_loss)
        result.log('training_loss', mean_loss, prog_bar=True)
        return result
    
    def _eval_step(self, batch_dict: dict) -> pl.EvalResult:
        loss, mse_loss, regularization_loss = self._get_batch_loss(batch_dict)
        assert tuple(loss.shape) == tuple(mse_loss.shape) == tuple(regularization_loss.shape)
        assert len(loss.shape) == 1 # batch_size
        result = pl.EvalResult()
        result.log('loss', loss)
        result.log('regularization_loss', regularization_loss)
        result.log('mse_loss', mse_loss)
        return result
    
    def _eval_epoch_end(self, step_result: pl.EvalResult, eval_type: Literal['validation', 'testing']) -> pl.EvalResult:
        loss = step_result.loss.mean()
        mse_loss = step_result.mse_loss.mean()
        regularization_loss = step_result.regularization_loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{eval_type}_loss', loss)
        result.log(f'{eval_type}_regularization_loss', regularization_loss)
        result.log(f'{eval_type}_mse_loss', mse_loss)
        return result
        
    def validation_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def validation_epoch_end(self, validation_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(validation_step_results, 'validation')

    def test_step(self, batch_dict: dict, _: int) -> pl.EvalResult:
        return self._eval_step(batch_dict)

    def test_epoch_end(self, test_step_results: pl.EvalResult) -> pl.EvalResult:
        return self._eval_epoch_end(test_step_results, 'testing')
    
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
            for hyperparameter_name in sorted(model.hparams.keys()):
                LOGGER.info(f'{hyperparameter_name}: {model.hparams[hyperparameter_name]:,}')
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

    @staticmethod
    @abstractmethod
    def checkpoint_directory_from_hyperparameters(**kwargs) -> str:
        pass
    
    @classmethod
    def train_model(cls, gpus: List[int], **hyperparameter_dict) -> float:
    
        checkpoint_dir = cls.checkpoint_directory_from_hyperparameters(**hyperparameter_dict)
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
            callbacks=[cls.PrintingCallback(checkpoint_callback)],
            max_epochs=hyperparameter_dict['number_of_epochs'],
            min_epochs=hyperparameter_dict['number_of_epochs'],
            gradient_clip_val=hyperparameter_dict.get('gradient_clip_val', 0),
            terminate_on_nan=True,
            gpus=gpus,
            distributed_backend='dp',
            deterministic=True,
            # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
            logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='cf_model'),
            checkpoint_callback=checkpoint_callback,
        )
    
        data_module = AnimeRatingDataModule(hyperparameter_dict['batch_size'], NUM_WORKERS)
        data_module.prepare_data()
        data_module.setup()
    
        model = cls(number_of_training_batches = len(data_module.training_dataloader), number_of_animes=data_module.number_of_animes, number_of_users=data_module.number_of_users, **hyperparameter_dict)
    
        trainer.fit(model, data_module)
        test_results = only_one(trainer.test(model, datamodule=data_module, verbose=False, ckpt_path=checkpoint_callback.best_model_path))
        best_validation_loss = checkpoint_callback.best_model_score.item()
        
        output_json_file_location = os.path.join(checkpoint_dir, 'result_summary.json')
        with open(output_json_file_location, 'w') as file_handle:
            json_dict = {
                'testing_loss': test_results['testing_loss'],
                'testing_regularization_loss': test_results['testing_regularization_loss'],
                'testing_mse_loss': test_results['testing_mse_loss'],
                
                'best_validation_loss': best_validation_loss,
                'best_validation_model_path': checkpoint_callback.best_model_path,
                
                'training_set_batch_size': data_module.training_dataloader.batch_size,
                'training_set_batch_count': len(data_module.training_dataloader),
                'validation_set_batch_size': data_module.validation_dataloader.batch_size,
                'validation_set_batch_count': len(data_module.validation_dataloader),
                'testing_set_batch_size': data_module.test_dataloader().batch_size,
                'testing_set_batch_count': len(data_module.test_dataloader()),
            }
            json_dict.update(hyperparameter_dict)
            json.dump(json_dict, file_handle, indent=4)
    
        return best_validation_loss

################
# Linear Model #
################

class LinearColaborativeFilteringModel(AbstractColaborativeFilteringModel):
    
    def __init__(
            self,
            learning_rate: float,
            number_of_epochs: int,
            number_of_training_batches: int,
            number_of_animes: int, 
            number_of_users: int, 
            embedding_size: int, 
            regularization_factor: float,
            dropout_probability: float,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            'learning_rate',
            'number_of_epochs',
            'number_of_training_batches',
            'number_of_animes',
            'number_of_users',
            'embedding_size',
            'regularization_factor',
            'dropout_probability',
        )
        
        self.anime_embedding_layer = nn.Embedding(self.hparams.number_of_animes, self.hparams.embedding_size)
        self.user_embedding_layer = nn.Embedding(self.hparams.number_of_users, self.hparams.embedding_size)
        self.dropout_layer = nn.Dropout(self.hparams.dropout_probability)

    def forward(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_id_indices: torch.Tensor = batch_dict['user_id_index']
        batch_size = user_id_indices.shape[0]
        assert tuple(user_id_indices.shape) == (batch_size,)
        anime_id_indices: torch.Tensor = batch_dict['anime_id_index']
        assert tuple(anime_id_indices.shape) == (batch_size,)

        user_embedding = self.user_embedding_layer(user_id_indices)
        assert tuple(user_embedding.shape) == (batch_size, self.hparams.embedding_size)
        anime_embedding = self.anime_embedding_layer(anime_id_indices)
        assert tuple(anime_embedding.shape) == (batch_size, self.hparams.embedding_size)
        
        get_norm = lambda batch_matrix: batch_matrix.mul(batch_matrix).sum(dim=1)
        regularization_loss = get_norm(user_embedding[:,:-1]) + get_norm(anime_embedding[:,:-1])
        assert tuple(regularization_loss.shape) == (batch_size,)
        assert regularization_loss.gt(0).all()
        
        user_embedding = self.dropout_layer(user_embedding)
        assert tuple(user_embedding.shape) == (batch_size, self.hparams.embedding_size)
        anime_embedding = self.dropout_layer(anime_embedding)
        assert tuple(anime_embedding.shape) == (batch_size, self.hparams.embedding_size)
        
        scores = user_embedding.mul(anime_embedding).sum(dim=1)
        assert tuple(scores.shape) == (batch_size,)
        
        return scores, regularization_loss

    @staticmethod
    def checkpoint_directory_from_hyperparameters(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, regularization_factor: float, dropout_probability: float) -> str:
        checkpoint_dir = f'./checkpoints/' \
            f'linear_cf_lr_{learning_rate:.5g}_' \
            f'epochs_{int(number_of_epochs)}_' \
            f'batch_{int(batch_size)}_' \
            f'gradient_clip_{gradient_clip_val:.3g}_' \
            f'embed_{int(embedding_size)}_' \
            f'regularization_{regularization_factor:.5g}_' \
            f'dropout_{dropout_probability:.5g}'
        return checkpoint_dir

############################
# Deep Concatenation Model #
############################

class DeepConcatenationColaborativeFilteringModel(AbstractColaborativeFilteringModel):
    
    def __init__(
            self,
            learning_rate: float,
            number_of_epochs: int,
            number_of_training_batches: int,
            number_of_animes: int, 
            number_of_users: int, 
            embedding_size: int, 
            dense_layer_count: int, 
            regularization_factor: float,
            dropout_probability: float, 
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            'learning_rate',
            'number_of_epochs',
            'number_of_training_batches',
            'number_of_animes',
            'number_of_users',
            'embedding_size',
            'dense_layer_count',
            'regularization_factor',
            'dropout_probability',
        )
        
        self.anime_embedding_layer = nn.Embedding(self.hparams.number_of_animes, self.hparams.embedding_size)
        self.user_embedding_layer = nn.Embedding(self.hparams.number_of_users, self.hparams.embedding_size)
        self.dropout_layer = nn.Dropout(self.hparams.dropout_probability)
        self.dense_layers = nn.Sequential(
            OrderedDict(
                sum(
                    [
                        [
                            (f'dense_layer_{i}', nn.Linear(self.hparams.embedding_size * 2, self.hparams.embedding_size * 2)),
                            (f'dropout_layer_{i}', nn.Dropout(self.hparams.dropout_probability)),
                            (f'activation_layer_{i}', nn.ReLU(True)),
                        ]
                        for i in range(self.hparams.dense_layer_count-1)
                    ],
                []) + [
                    (f'dense_layer_{self.hparams.dense_layer_count-1}', nn.Linear(self.hparams.embedding_size * 2, 1)),
                    (f'activation_layer_{self.hparams.dense_layer_count-1}', nn.Sigmoid()),
                ])
        )
    
    def forward(self, batch_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_id_indices: torch.Tensor = batch_dict['user_id_index']
        batch_size = user_id_indices.shape[0]
        assert tuple(user_id_indices.shape) == (batch_size,)
        anime_id_indices: torch.Tensor = batch_dict['anime_id_index']
        assert tuple(anime_id_indices.shape) == (batch_size,)

        user_embedding = self.user_embedding_layer(user_id_indices)
        assert tuple(user_embedding.shape) == (batch_size, self.hparams.embedding_size)
        anime_embedding = self.anime_embedding_layer(anime_id_indices)
        assert tuple(anime_embedding.shape) == (batch_size, self.hparams.embedding_size)
        
        get_norm = lambda batch_matrix: batch_matrix.mul(batch_matrix).sum(dim=1)
        user_regularization_loss = get_norm(user_embedding[:,:-1])
        anime_regularization_loss = get_norm(anime_embedding[:,:-1])
        regularization_loss = user_regularization_loss + anime_regularization_loss
        # assert regularization_loss.gt(0).all() # the model magicaly works even with embeddings near zero
        
        user_embedding = self.dropout_layer(user_embedding)
        assert tuple(user_embedding.shape) == (batch_size, self.hparams.embedding_size)
        anime_embedding = self.dropout_layer(anime_embedding)
        assert tuple(anime_embedding.shape) == (batch_size, self.hparams.embedding_size)

        concatenated_embedding = torch.cat([user_embedding, anime_embedding], dim=1)
        assert tuple(concatenated_embedding.shape) == (batch_size, self.hparams.embedding_size*2)
        
        scores = self.dense_layers(concatenated_embedding)
        scores = scores.view(-1)
        scores = scores * 10
        assert tuple(scores.shape) == (batch_size,)
        
        return scores, regularization_loss

    @staticmethod
    def checkpoint_directory_from_hyperparameters(learning_rate: float, number_of_epochs: int, batch_size: int, gradient_clip_val: float, embedding_size: int, dense_layer_count: int, regularization_factor: float, dropout_probability: float) -> str:
        checkpoint_dir = f'./checkpoints/' \
            f'concat_cf_lr_{learning_rate:.5g}_' \
            f'epochs_{int(number_of_epochs)}_' \
            f'batch_{int(batch_size)}_' \
            f'gradient_clip_{gradient_clip_val:.3g}_' \
            f'embed_{int(embedding_size)}_' \
            f'dense_layers_{int(dense_layer_count)}_' \
            f'regularization_{regularization_factor:.5g}_' \
            f'dropout_{dropout_probability:.5g}'
        return checkpoint_dir

if __name__ == '__main__':
    print('This module contains collaborative filtering models to learn representations of users and animes.')
