import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score

"""
noisy detection task
"""

def noisy_detection(data_values, noisy_index, num_classes=2):
    n_noisy = len(noisy_index)
    index_of_small_values = np.argsort(data_values)[: n_noisy]
    recall = len(np.intersect1d(index_of_small_values, noisy_index)) / n_noisy

    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(data_values.reshape(-1, 1))

    # using kmeans label
    guess_index = np.where(kmeans.labels_ == np.argmin(kmeans.cluster_centers_))[0]
    f1_kmeans_label = f1_score(kmeans.labels_, np.argmin(kmeans.cluster_centers_))

    return recall, f1_kmeans_label


def point_removal(X, y, X_test, y_test, select_method: str, value_list: np.array=None, problem="clf"):
    n_sample = len(X)
    if select_method == 'random':
        sorted_value_list = np.random.permutation(n_sample)
    elif select_method == 'ascending':  # ascending order. low to high.
        sorted_value_list = np.argsort(value_list)
    else:  # descending order. high to low.
        sorted_value_list = np.argsort(value_list)[::-1]

    accuracy_list = []
    n_period = min(n_sample // 200, 5)  # we add 0.5% at each time
    for percentile in range(0, n_sample, n_period):
        """
        We repeatedly remove 5% of entire data points at each step.
        The data points whose value belongs to the lowest group are removed first.
        The larger, the better
        """
        sorted_value_list_tmp = sorted_value_list[percentile:]
        if problem == "clf":
            try:
                clf = LogisticRegression()
                clf.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score = clf.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score = np.mean(np.mean(y[sorted_value_list_tmp]) == y_test)
        else:
            try:
                model = LinearRegression()
                model.fit(X[sorted_value_list_tmp], y[sorted_value_list_tmp])
                model_score = model.score(X_test, y_test)
            except:
                # if y[sorted_value_list_tmp] only has one class
                model_score = 0

        accuracy_list.append(model_score)

    return accuracy_list


def remove_high_low(
    dve_out, eval_model, x_train, y_train,
    x_valid, y_valid, x_test, y_test,
    perf_metric='rmspe', plot=True
):
  """Evaluates performance after removing a portion of high/low valued samples.
  Args:
    dve_out: data values
    eval_model: evaluation model (object)
    x_train: training features
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    x_test: testing features
    y_test: testing labels
    perf_metric: 'auc', 'accuracy', or 'rmspe'
    plot: print plot or not
  Returns:
    output_perf: Prediction performances after removing a portion of high
                 or low valued samples.
  """

    x_train = np.asarray(x_train)
    y_train = np.reshape(np.asarray(y_train), [len(y_train),])
    x_valid = np.asarray(x_valid)
    y_valid = np.reshape(np.asarray(y_valid), [len(y_valid),])
    x_test = np.asarray(x_test)
    y_test = np.reshape(np.asarray(y_test), [len(y_test),])

    # Sorts samples by data values
    num_bins = 20  # Per 100/20 percentile
    sort_idx = np.argsort(dve_out)
    n_sort_idx = np.argsort(-dve_out)

    # Output Initialization
    if perf_metric in ['auc', 'accuracy']:
        temp_output = np.zeros([2 * num_bins, 2])
    elif perf_metric == 'rmspe':
        temp_output = np.ones([2 * num_bins, 2])

    # For each percentile bin
    for itt in range(num_bins):

        # 1. Remove least valuable samples first
        new_x_train = x_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:

        eval_model.fit(new_x_train, new_y_train)

        if perf_metric == 'auc':
            y_valid_hat = eval_model.predict_proba(x_valid)[:, 1]
            y_test_hat = eval_model.predict_proba(x_test)[:, 1]

            temp_output[itt, 0] = auroc(y_valid, y_valid_hat)
            temp_output[itt, 1] = auroc(y_test, y_test_hat)

        elif perf_metric == 'accuracy':
            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[itt, 0] = metrics.accuracy_score(y_valid,
                                                        np.argmax(y_valid_hat,
                                                                axis=1))
            temp_output[itt, 1] = metrics.accuracy_score(y_test,
                                                        np.argmax(y_test_hat,
                                                                axis=1))
        elif perf_metric == 'rmspe':
            y_valid_hat = eval_model.predict(x_valid)
            y_test_hat = eval_model.predict(x_test)

            temp_output[itt, 0] = rmspe(y_valid, y_valid_hat)
            temp_output[itt, 1] = rmspe(y_test, y_test_hat)

        # 2. Remove most valuable samples first
        new_x_train = x_train[n_sort_idx[int(itt*len(x_train[:, 0])/num_bins):], :]
        new_y_train = y_train[n_sort_idx[int(itt*len(x_train[:, 0])/num_bins):]]

        if len(np.unique(new_y_train)) > 1:

        eval_model.fit(new_x_train, new_y_train)

        if perf_metric == 'auc':
            y_valid_hat = eval_model.predict_proba(x_valid)[:, 1]
            y_test_hat = eval_model.predict_proba(x_test)[:, 1]

            temp_output[num_bins + itt, 0] = auroc(y_valid, y_valid_hat)
            temp_output[num_bins + itt, 1] = auroc(y_test, y_test_hat)

        elif perf_metric == 'accuracy':
            y_valid_hat = eval_model.predict_proba(x_valid)
            y_test_hat = eval_model.predict_proba(x_test)

            temp_output[num_bins + itt, 0] = \
                metrics.accuracy_score(y_valid, np.argmax(y_valid_hat, axis=1))
            temp_output[num_bins + itt, 1] = \
                metrics.accuracy_score(y_test, np.argmax(y_test_hat, axis=1))

        elif perf_metric == 'rmspe':
            y_valid_hat = eval_model.predict(x_valid)
            y_test_hat = eval_model.predict(x_test)

            temp_output[num_bins + itt, 0] = rmspe(y_valid, y_valid_hat)
            temp_output[num_bins + itt, 1] = rmspe(y_test, y_test_hat)

    # Plot graphs
    if plot:

        # Defines x-axis
        num_x = int(num_bins/2 + 1)
        x = [a*(1.0/num_bins) for a in range(num_x)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, temp_output[:num_x, 1], 'o-')
        plt.plot(x, temp_output[num_bins:(num_bins+num_x), 1], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel('Accuracy', size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                prop={'size': 16})
        plt.title('Remove High/Low Valued Samples', size=16)

        plt.show()

    return temp_output


def noisy_detection_experiment(value_dict, noisy_index):
    noisy_score_dict = dict()
    for key in value_dict.keys():
        noisy_score_dict[key] = noisy_detection_core(value_dict[key], noisy_index)

    noisy_dict = {"Meta_Data": ["Recall", "Kmeans_label"], "Results": noisy_score_dict}
    return noisy_dict


def discover_corrupted_sample(dve_out, noise_idx, noise_rate, plot=True):
  """Reports True Positive Rate (TPR) of corrupted label discovery.
  Args:
    dve_out: data values
    noise_idx: noise index
    noise_rate: the ratio of noisy samples
    plot: print plot or not
  Returns:
    output_perf: True positive rate (TPR) of corrupted label discovery
                 (per 5 percentiles)
  """

  # Sorts samples by data values
  num_bins = 20  # Per 100/20 percentile
  sort_idx = np.argsort(dve_out)

  # Output initialization
  output_perf = np.zeros([num_bins,])

  # For each percentile
  for itt in range(num_bins):
    # from low to high data values
    output_perf[itt] = len(np.intersect1d(sort_idx[:int((itt+1)* \
                              len(dve_out)/num_bins)], noise_idx)) \
                              / len(noise_idx)

  # Plot corrupted label discovery graphs
  if plot:

    # Defines x-axis
    num_x = int(num_bins/2 + 1)
    x = [a*(1.0/num_bins) for a in range(num_x)]

    # Corrupted label discovery results (dvrl, optimal, random)
    y_dvrl = np.concatenate((np.zeros(1), output_perf[:(num_x-1)]))
    y_opt = [min([a*((1.0/num_bins)/noise_rate), 1]) for a in range(num_x)]
    y_random = x

    plt.figure(figsize=(6, 7.5))
    plt.plot(x, y_dvrl, 'o-')
    plt.plot(x, y_opt, '--')
    plt.plot(x, y_random, ':')
    plt.xlabel('Fraction of data Inspected', size=16)
    plt.ylabel('Fraction of discovered corrupted samples', size=16)
    plt.legend(['DVRL', 'Optimal', 'Random'], prop={'size': 16})
    plt.title('Corrupted Sample Discovery', size=16)
    plt.show()

  # Returns True Positive Rate of corrupted label discovery
  return output_perf

def point_removal_experiment(value_dict, X, y, X_test, y_test, problem="clf"):
    removal_ascending_dict, removal_descending_dict = dict(), dict()
    for key in value_dict.keys():
        removal_ascending_dict[key] = point_removal_core(
            X, y, X_test, y_test, value_dict[key], ascending=True, problem=problem
        )
        removal_descending_dict[key] = point_removal_core(
            X, y, X_test, y_test, value_dict[key], ascending=False, problem=problem
        )
    random_array = point_removal_core(X, y, X_test, y_test, "Random", problem=problem)
    removal_ascending_dict["Random"] = random_array
    removal_descending_dict["Random"] = random_array
    return {"ascending": removal_ascending_dict, "descending": removal_descending_dict}


