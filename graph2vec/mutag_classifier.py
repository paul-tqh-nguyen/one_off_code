
'''

Sections:
* Imports
* Globals

'''

# @todo update docstring

###########
# Imports #
###########

from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from torch import nn
from collections import OrderedDict
from typing import Tuple

from global_values import *
from misc_utilities import *

# @todo make sure these imports are used

###########
# Globals #
###########

BCE_LOSS = torch.nn.BCELoss()

#####################
# Graph Data Module #
#####################

class MUTAGDataset(data.Dataset):
    def __init__(self, graph_id_to_graph_label: Dict[int, int]):
        self.graph_id_to_graph_label = graph_id_to_graph_label
        
    def __getitem__(self, graph_id: int):
        return self.graph_id_to_graph_label[graph_id]
    
    def __len__(self):
        return len(self.df)

class MUTAGDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, graph_id_to_graph_label: Dict[int, int]): 
        self.batch_size = batch_size
        self.graph_id_to_graph_label = graph_id_to_graph_label
        
    def prepare_data(self) -> None:
        return
    
    def setup(self) -> None:
        
        graph_ids = graph_id_to_graph_label.keys()
        training_graph_ids, testing_graph_ids = train_test_split(graph_ids, test_size=0.30, random_state=RANDOM_SEED)
        validation_graph_ids, testing_graph_ids = train_test_split(testing_graph_ids, test_size=0.5, random_state=RANDOM_SEED)

        training_graph_id_to_graph_label = {}
        validation_graph_id_to_graph_label = {}
        testing_graph_id_to_graph_label = {}

        for graph_id, graph_label in graph_id_to_graph_label.items():
            if graph_id in training_graph_ids:
                training_graph_id_to_graph_label[graph_id] = graph_label
            elif graph_id in validation_graph_ids:
                validation_graph_id_to_graph_label[graph_id] = graph_label
            elif graph_id in testing_graph_ids:
                testing_graph_id_to_graph_label[graph_id] = graph_label
            else:
                raise ValueError(f'{graph_id} not in any split.')
    
        
        training_dataset = MUTAGDataset(training_graph_id_to_graph_label)
        validation_dataset = MUTAGDataset(validation_graph_id_to_graph_label)
        testing_dataset = MUTAGDataset(testing_graph_id_to_graph_label)

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/408 forces us to use shuffle and drop_last in training
        self.training_dataloader = data.DataLoader(training_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)
        self.validation_dataloader = data.DataLoader(validation_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
        self.testing_dataloader = data.DataLoader(testing_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
        
        return
    
    def train_dataloader(self) -> data.DataLoader:
        return self.training_dataloader

    def val_dataloader(self) -> data.DataLoader:
        return self.validation_dataloader

    def test_dataloader(self) -> data.DataLoader:
        return self.testing_dataloader

##########################
# MUTAG Classifier Model #
##########################

class MUTAGClassifier(pl.LightningModule):

    hyperparameter_names = (
        'batch_size', 
        'graph2vec_trial_index',
        'number_of_layers',
        'gradient_clip_val',
        'dropout_probability',
    )
    
    def __init__(self, graph_id_to_graph_embeddings: VectorDict, graph_id_to_graph_label: Dict[int, int], batch_size: int, graph2vec_trial_index: float, number_of_layers: int, gradient_clip_val: float, dropout_probability: float):
        super().__init__()
        self.save_hyperparameters(**self.__class__.hyperparameter_names)
        
        graph2vec_study_df = optuna.create_study(study_name=GRAPH2VEC_STUDY_NAME, storage=GRAPH2VEC_DB_URL, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner()).trials_dataframe()
        graph_vector_size = graph2vec_study_df.iloc[self.hparams.graph2vec_trial_index]['params_dimensions']

        self.linear_layers = nn.Sequential(
            OrderedDict( # @todo try decreasing the linear layer sizes
                sum(
                    [
                        [
                            (f'dense_layer_{i}', nn.Linear(graph_vector_size, graph_vector_size)),
                            (f'dropout_layer_{i}', nn.Dropout(self.hparams.dropout_probability)),
                            (f'activation_layer_{i}', nn.ReLU(True)),
                        ] for i in range(number_of_layers)
                    ], [])
            )
        )
        self.prediction_layers = nn.Sequential(
            OrderedDict([
                (f'reduction_layer', nn.Linear(graph_vector_size, 1)),
                (f'activation_layer', nn.Tanh()),
            ])
        )

        self.data_module = MUTAGDataModule(hyperparameter_dict['batch_size'], graph_id_to_graph_label)
        self.data_module.prepare_data()
        self.data_module.setup()

    def forward(self, batch: torch.Tensor) -> torch.Tensor: # @todo sweep all return signatures
        batch_size = batch.shape[0]
        graph_vector_size = self.linear_layers.dense_layer_0.in_features
        assert tuple(batch.shape) == (batch_size, graph_vector_size)

        transformed_batch = self.linear_layers(batch)
        assert tuple(transformed_batch.shape) == (batch_size, graph_vector_size)
        
        predictions = self.prediction_layers(transformed_batch).squeeze(1)
        assert tuple(predictions.shape) == (batch_size,)
        
        return predictions

    def backward(self, _trainer: pl.Trainer, loss: torch.Tensor, _optimizer: torch.optim.Optimizer, _optimizer_idx: int) -> None:
        del _trainer, _optimizer, _optimizer_idx
        loss.mean().backward() # @todo do we need this mean call here?
        return

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        # @todo use Nadam and explore its parameters in the hyperparameter search
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate) # @todo print out the learning rate to make sure this works
        return {'optimizer': optimizer}
        
    def _get_batch_loss(self, batch_dict: dict) -> torch.Tensor:
        batch = batch_dict['graph_embedding']
        target_predictions = batch_dict['target']
        batch_size = only_one(target_predictions.shape)
        graph_vector_size = self.linear_layers.dense_layer_0.in_features
        assert tuple(batch.shape) == (batch_size, graph_vector_size)
        assert tuple(target_predictions.shape) == (batch_size,)
        
        predictions = self(batch)
        assert tuple(predictions.shape) == (batch_size,)
        bce_loss = BCE_LOSS(predictions, target_predictions)
        return bce_loss
    
    def training_step(self, batch_dict: dict, _: int) -> pl.TrainResult:
        loss = self._get_batch_loss(batch_dict)
        result = pl.TrainResult(minimize=loss)
        return result

    def training_step_end(self, training_step_result: pl.TrainResult) -> pl.TrainResult:
        assert len(training_step_result.minimize.shape) == 1
        mean_loss = training_step_result.minimize.mean()
        result = pl.TrainResult(minimize=mean_loss)
        result.log('training_loss', mean_loss, prog_bar=True)
        return result
    
    def _eval_step(self, batch_dict: dict) -> pl.EvalResult:
        loss = self._get_batch_loss(batch_dict)
        assert len(loss.shape) == 1 # batch_size
        result = pl.EvalResult()
        result.log('loss', loss)
        return result
    
    def _eval_epoch_end(self, step_result: pl.EvalResult, eval_type: Literal['validation', 'testing']) -> pl.EvalResult:
        loss = step_result.loss.mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{eval_type}_loss', loss)
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
    def checkpoint_directory_from_hyperparameters(batch_size: int, graph2vec_trial_index: float, number_of_layers: int, gradient_clip_val: float, dropout_probability: float) -> str:
        checkpoint_dir = f'./checkpoints/' \
            f'batch_{int(batch_size)}_' \
            f'graph2vec_trial_index_{int(graph2vec_trial_index)}_' \
            f'number_of_layers_{int(number_of_layers)}_' \
            f'gradient_clip_{int(gradient_clip_val)}_' \
            f'dropout_{dropout_probability:.5g}'
        return checkpoint_dir

    @classmethod
    def train_model(cls, gpus: List[int], **model_initialization_args) -> float:
        
        hyperparameter_dict = {
            hyperparameter_name: hyperparameter_value
            for hyperparameter_name, hyperparameter_value in model_initialization_args.items
            if hyperparameter_name in cls.hyperparameter_names
        }
        
        checkpoint_dir = cls.checkpoint_directory_from_hyperparameters(**model_initialization_args)
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
            auto_lr_find=True,
            early_stop_callback=EarlyStopping(
                monitor='val_accuracy',
                min_delta=1.00, # @todo change this
                patience=3,
                verbose=True, # @todo turn this off
                mode='min'
            ),
            gradient_clip_val=model_initialization_args.get('gradient_clip_val', 0),
            terminate_on_nan=True,
            gpus=gpus,
            distributed_backend='dp',
            deterministic=True,
            # precision=16, # not supported for data parallel (e.g. multiple GPUs) https://github.com/NVIDIA/apex/issues/227
            logger=pl.loggers.TensorBoardLogger(checkpoint_dir, name='cf_model'),
            checkpoint_callback=checkpoint_callback,
        )
    
        model = cls(**model_initialization_args)
    
        trainer.fit(model, self.data_module)
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
