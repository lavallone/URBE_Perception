import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks import QuantizationAwareTraining

def train_model(data, model, experiment_name, patience, metric_to_monitor, mode, epochs):
    logger =  WandbLogger()
    logger.experiment.watch(model, log = None, log_freq = 100000)
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor, mode=mode, min_delta=0.00, patience=patience, verbose=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor=metric_to_monitor, mode=mode, dirpath="models",
        filename=experiment_name +
        "-{epoch:02d}-{mAP_50:.4f}", verbose=True)
    if model.hparams.reduce_inference == True:
        amount = 0.1 if model.hparams.first_out == 48 else 0.05 # if the model is medium or nano we change this value
        pruning = ModelPruning("l1_unstructured", amount=amount, use_lottery_ticket_hypothesis=True) # the pruning is computed each epoch!
        quantization = QuantizationAwareTraining()
        callbacks = [early_stop_callback, checkpoint_callback, pruning, quantization]
    else:
        callbacks = [early_stop_callback, checkpoint_callback]
    # the trainer collect all the useful informations so far for the training
    n_gpus = 1 if torch.cuda.is_available() else 0
    if model.hparams.resume_from_checkpoint is not None:
        trainer = pl.Trainer(
            logger=logger, max_epochs=epochs, log_every_n_steps=1, gpus=n_gpus,
            callbacks=callbacks, precision = model.hparams.precision,
            num_sanity_val_steps=0, resume_from_checkpoint=model.hparams.resume_from_checkpoint
            )
    else:
        trainer = pl.Trainer(
            logger=logger, max_epochs=epochs, log_every_n_steps=1, gpus=n_gpus,
            callbacks=callbacks, precision = model.hparams.precision,
            num_sanity_val_steps=0,
            )
    trainer.fit(model, data)
    return trainer
