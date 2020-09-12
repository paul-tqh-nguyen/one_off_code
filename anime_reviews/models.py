#!/usr/bin/python3 -OO

'''

This module contains a linear collaborative filteringg model to learn representations of users and animes.

Sections:
* Imports
* Models

'''

###########
# Imports #
###########

from typing import Tuple
from typing_extensions import Literal

import torch
from torch import nn
import transformers
import pytorch_lightning as pl

from misc_utilities import *
from global_values import *

##########
# Models #
##########

MSE_LOSS = nn.MSELoss(reduction='none')

class LinearColaborativeFilteringModel(pl.LightningModule):
    
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

    def forward(self, batch_dict: dict):
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

if __name__ == '__main__':
    print('This module contains a linear collaborative filtering model to learn representations of users and animes.')
