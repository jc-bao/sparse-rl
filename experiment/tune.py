import optuna
import numpy as np


def objective(triail):
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

study = optuna.create_study(
    sampler=sampler,
    pruner=pruner,
    storage=self.storage,
    study_name=self.study_name,
    load_if_exists=True,
    direction="maximize",
)

study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)