import random, argh
import numpy as np
from run_experiment import run_experiment_core

parser = argh.ArghParser()


def _set_seed(config):
    print(f"Config: {config}")
    random.seed(config["run_id"])
    np.random.seed(config["run_id"])


def main(config):
    _set_seed(config)
    run_experiment_core(config)
    print("Done!")


parser.add_commands([main])
if __name__ == "__main__":
    parser.dispatch()
