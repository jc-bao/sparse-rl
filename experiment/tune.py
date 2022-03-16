import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import numpy as np
from .train import launch


def objective(trial):
    launch()
    evaluation_score = 1
    return evaluation_score

def sampler(trial: optuna.Trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096, 8192])
    n_batches = trial.suggest_categorical("train_freq", 10*(np.arange(10)+1))

    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_batches": n_batches
    }

    return hyperparams

if __name__=='__main__':
    wandbc = WeightsAndBiasesCallback(wandb_kwargs={'project': 'debug'})
    study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.SuccessiveHalvingPruner(reduction_factor=4, min_early_stopping_rate=0),
        study_name='handover',
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(objective, n_trials=16, n_jobs=4, callbacks=[wandbc])