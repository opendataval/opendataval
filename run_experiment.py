import datasets
from data_valuation import DataValuation

import pickle
import numpy as np


def run_experiment_core(config):
    """
    Initialize variables
    """
    runpath = config["runpath"]
    problem = config["problem"]
    dataset = config["dataset"]
    dargs_list = config["dargs_list"]

    if dataset != "gaussian":
        loo_run = True
        betashap_run = True
        AME_run = True
        lasso_run = True
        boosting_run = True
        treeshap_run = True
        removal_run = True
        simple_run = False
    else:
        loo_run = False
        betashap_run = False
        AME_run = False
        lasso_run = False
        boosting_run = False
        treeshap_run = False
        removal_run = False
        simple_run = False

    for dargs_ind in range(len(dargs_list)):
        # Load dataset
        dargs = dargs_list[dargs_ind]
        (X, y), (X_val, y_val), (X_test, y_test), noisy_index = datasets.load_data(
            problem, dataset, **dargs
        )

        # instantiate data valuation engine
        data_valuation_engine = DataValuation(
            X=X, y=y, X_val=X_val, y_val=y_val, problem=problem, dargs=dargs
        )

        # evaluate baseline peroformance
        data_valuation_engine.evaluate_baseline_models(X_test, y_test)

        # compute data values
        data_valuation_engine.compute_data_shap(
            loo_run=loo_run, betashap_run=betashap_run
        )
        data_valuation_engine.compute_feature_shap(
            AME_run=AME_run,
            lasso_run=lasso_run,
            boosting_run=boosting_run,
            treeshap_run=treeshap_run,
            simple_run=simple_run,
        )

        # evaluate the quality of data values
        data_valuation_engine.evaluate_data_values(
            noisy_index, X_test, y_test, removal_run=removal_run
        )

        # save results
        data_valuation_engine.save_results(runpath, dataset, dargs_ind, noisy_index)

        if len(X) == 1000 and dargs["run_id"] <= 2:
            run_id = dargs["run_id"]
            data_dict = {"X": X, "y": y, "X_test": X_test, "y_tset": y_test}
            with open(runpath + f"/dataset_{run_id}_{dargs_ind}.pkl", "wb") as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Data is saved! path: {runpath}, run_id: {run_id}.", flush=True)

        del X, y, X_val, y_val, X_test, y_test
        del data_valuation_engine
