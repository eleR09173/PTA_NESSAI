# COSI' COMPILA SOLO SE IN root/PTA_NESSAI/OptunaTuning

import os
import torch
import optuna
from pathlib import Path

from multi_sampler_runner import run_multi_pulsar_sampler

# Create output directory
output_base = Path(__file__).resolve().parent / "runs"
os.makedirs(output_base, exist_ok=True)

def objective(trial):
    max_threads = os.cpu_count()

    config = {
        "n_blocks": trial.suggest_int("n_blocks", 4, 10),
        "n_layers": trial.suggest_int("n_layers", 4, 10),
        "n_neurons": trial.suggest_int("n_neurons", 8, 32),
        "pytorch_threads": trial.suggest_int("pytorch_threads", 1, max_threads),
        "n_pool": 1  # multiprocessing is disabled for now
    }

    torch.set_num_threads(config["pytorch_threads"])

    output_path = output_base / f"trial_{trial.number}"
    try:
        total_time = run_multi_pulsar_sampler(settings=config, output_path=output_path, seed=trial.number)
        return total_time
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")

    for i in range(10):
        try:
            study.optimize(objective, n_trials=1, n_jobs=1)
        except Exception as e:
            print(f"Trial {i} crashed: {e}")

    print("Best parameters:")
    print(study.best_params)
