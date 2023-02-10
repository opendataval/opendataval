import numpy as np
import copy, socket, getpass


def generate_config(
    expno_name,
    dataset="covertype",
    problem="clf",
    n_test=3000,
    model_family="Tree",
    n_runs=50,
):
    """
    This function creates experiment configurations.
    """

    # Experiment configuration
    exp = dict()
    exp["expno"] = expno_name
    exp["n_runs"] = n_runs

    # Run configuration
    run_temp = dict()
    run_temp["problem"] = problem
    run_temp["dataset"] = dataset

    runs = []
    if expno_name != "000CR":
        if problem == "clf":
            for run_id in range(n_runs):
                run = copy.deepcopy(run_temp)
                run["run_id"] = run_id
                dargs_list = []
                for n_data_to_be_valued in [1000, 10000]:
                    for is_noisy in [0.1]:
                        for n_trees in [800, 3200]:
                            dargs_list.append(
                                {
                                    "n_data_to_be_valued": n_data_to_be_valued,
                                    "n_val": (n_data_to_be_valued // 10),
                                    "n_test": n_test,
                                    "n_trees": n_trees,
                                    "clf_path": exp["clf_path"],
                                    "openml_clf_path": exp["openml_clf_path"],
                                    "openml_reg_path": exp["openml_reg_path"],
                                    "is_noisy": is_noisy,
                                    "model_family": model_family,
                                    "run_id": run_id,
                                }
                            )
                run["dargs_list"] = dargs_list
                runs.append(run)
        elif problem == "reg":
            for run_id in range(n_runs):
                run = copy.deepcopy(run_temp)
                run["run_id"] = run_id
                dargs_list = []
                for n_data_to_be_valued in [1000, 10000]:
                    for n_trees in [800]:
                        dargs_list.append(
                            {
                                "n_data_to_be_valued": n_data_to_be_valued,
                                "n_val": (n_data_to_be_valued // 10),
                                "n_test": n_test,
                                "n_trees": n_trees,
                                "clf_path": "clf_path",
                                "openml_clf_path": "openml_clf_path",
                                "openml_reg_path": "openml_reg_path",
                                "is_noisy": 0,
                                "model_family": model_family,
                                "run_id": run_id,
                            }
                        )
                run["dargs_list"] = dargs_list
                runs.append(run)
        else:
            assert False, f"Check Problem: {problem}"

    else:
        for run_id in range(n_runs):
            run = copy.deepcopy(run_temp)
            run["run_id"] = run_id
            dargs_list = []
            is_noisy = 0.1
            n_trees = 800
            for n_data_to_be_valued in [10000, 25000, 50000, 100000, 250000, 500000]:
                for input_dim in [10, 100]:
                    dargs_list.append(
                        {
                            "n_data_to_be_valued": n_data_to_be_valued,
                            "n_val": (n_data_to_be_valued // 10),
                            "n_test": n_test,
                            "n_trees": n_trees,
                            "clf_path": exp["clf_path"],
                            "openml_path": exp["openml_path"],
                            "is_noisy": is_noisy,
                            "model_family": model_family,
                            "input_dim": input_dim,
                            "run_id": run_id,
                        }
                    )
            run["dargs_list"] = dargs_list
            runs.append(run)

    return exp, runs


"""
Classification
"""


def config000CR():
    exp, runs = generate_config(
        expno_name="000CR", problem="clf", dataset="gaussian", n_runs=5
    )
    return exp, runs


def config001CR():
    exp, runs = generate_config(expno_name="001CR", problem="clf", dataset="pol")
    return exp, runs


def config002CR():
    exp, runs = generate_config(expno_name="002CR", problem="clf", dataset="jannis")
    return exp, runs


def config003CR():
    exp, runs = generate_config(expno_name="003CR", problem="clf", dataset="lawschool")
    return exp, runs


def config004CR():
    exp, runs = generate_config(expno_name="004CR", problem="clf", dataset="fried")
    return exp, runs


def config005CR():
    exp, runs = generate_config(
        expno_name="005CR", problem="clf", dataset="vehicle_sensIT"
    )
    return exp, runs


def config006CR():
    exp, runs = generate_config(
        expno_name="006CR", problem="clf", dataset="electricity"
    )
    return exp, runs


def config007CR():
    exp, runs = generate_config(expno_name="007CR", problem="clf", dataset="2dplanes")
    return exp, runs


def config008CR():
    exp, runs = generate_config(expno_name="008CR", problem="clf", dataset="creditcard")
    return exp, runs


def config009CR():
    exp, runs = generate_config(expno_name="009CR", problem="clf", dataset="covertype")
    return exp, runs


def config010CR():
    exp, runs = generate_config(expno_name="010CR", problem="clf", dataset="nomao")
    return exp, runs


def config011CR():
    exp, runs = generate_config(
        expno_name="011CR", problem="clf", dataset="webdata_wXa"
    )
    return exp, runs


def config012CR():
    exp, runs = generate_config(expno_name="012CR", problem="clf", dataset="MiniBooNE")
    return exp, runs
