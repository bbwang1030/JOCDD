import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from page_hinkley import PageHinkley
import pandas as pd
from pyoselm.oselm import OSELMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from IPCD import IPCD
import time
import random

# Data generation
np.random.seed(2)
n = 100000
bins = 50
n_i = round(n / bins)
d = 10

window_size = [250, 500, 750, 1000, 1250]
threshold = [5]
delta = 0.01
burn_in = 20
rate = [0, 0.001, 0.002, 0.005, 0.01]
# rate = [0]
eta = 1 / 500

epsilon = np.random.normal(0, 0.001, n).reshape(-1, 1)
np.random.seed(3)
# Generate random numbers following uniform distribution
X = np.random.uniform(0.2, 0.5, (n, d))


# Define generic drift detection function
def drift_detection_with_ph(model, X, Y, window_size, delta, threshold, burn_in):
    ph_detector = PageHinkley(delta=delta, threshold=threshold, burn_in=burn_in, direction="positive")
    reference_window_X = X[:window_size]
    reference_window_Y = Y[:window_size]
    model.fit(reference_window_X, reference_window_Y.ravel())

    drift_points = []
    for i in range(window_size, len(X)):
        test_window_X = X[i - window_size:i]
        test_window_Y = Y[i - window_size:i]

        new_X = X[i - window_size].reshape(1, -1)
        new_Y = Y[i - window_size]

        if ph_detector.drift_state == "drift":
            reference_window_X = test_window_X
            reference_window_Y = test_window_Y
            model.fit(reference_window_X, reference_window_Y.ravel())
            ph_detector.reset()

        y_pred = model.predict(new_X)
        residual = abs(new_Y - y_pred[0])
        ph_detector.update(np.array([residual]))
        if ph_detector.drift_state == "drift":
            drift_points.append(i - window_size + 1)

    return drift_points


plt.rcParams['figure.figsize'] = (8, 10)
linestyles = ['-.', '--', ':', '-.', '--', '-', '-']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#FFB6C1', '#9acd32', '#eee8aa', '#8470ff', '#625b57', '#87cefa', '#f44336']

# Simulate true drift points
true_drift_points = [i for i in range(2000, len(X), 2000)]
F1score = np.ones((len(rate), len(window_size), 8)) * (-1)
for num_r, r in enumerate(rate):
    # Generate data
    all_Y = np.array([])
    for i in range(bins):
        np.random.seed(i)
        # Generate random integers
        random_integers = np.random.rand(d)
        beta = np.array([
            [random_integers[0]],
            [random_integers[1]],
            [random_integers[2]],
            [random_integers[3]],
            [random_integers[4]],
            [random_integers[5]],
            [random_integers[6]],
            [random_integers[7]],
            [random_integers[8]],
            [random_integers[9]]
        ])
        Y = np.dot(X[i * n_i:(i + 1) * n_i, :], beta) + epsilon[i * n_i:(i + 1) * n_i]

        # Generate outliers
        for outlier in range(int(n * r / bins)):
            mean = np.random.uniform(low=3, high=5, size=1)
            std_dev = np.random.uniform(low=0, high=1, size=1)
            random_number = np.random.normal(mean, std_dev, size=1)
            if np.random.rand() < 0.5:
                random_number = np.random.normal((-mean), std_dev, size=1)

            outlier_index = int(n / (bins * (int(n * r / bins) + 1)) * (outlier + 1))
            if outlier_index < len(Y):
                Y[int(outlier_index)] = Y[int(outlier_index)] + random_number[0]

            for i in range(50):
                outlier_list.append(2000 * i + outlier_index)

        # Concatenate Y to all_Y
        if all_Y.size == 0:
            all_Y = Y
        else:
            all_Y = np.concatenate((all_Y, Y))

    # ThreThold loop
    for num_thres, thres in enumerate(threshold):
        # Window size loop
        for num_window, window in enumerate(window_size):
            # Model dictionary
            models = {
                "Lasso": Lasso(alpha=0.01, max_iter=1000),
                "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "Bagging": BaggingRegressor(random_state=42),
                "OSELM": OSELMRegressor(n_hidden=100),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                              random_state=42),
                "Linear": LinearRegression(),
                "Ipod": IPCD(X=X, Y=all_Y, window_size=window, threshold=thres, delta=delta, eta=eta, burn_in=burn_in)
            }

            # Loop through models
            results_list = []
            time_list = []
            for num, (name, model) in enumerate(models.items()):
                start_time = time.time()
                print(f"Processing model: {name}")
                if name != "Ipod":
                    drift_points = drift_detection_with_ph(model, X, all_Y, window, delta, thres, burn_in)
                elif name == "Ipod":
                    ipcd = IPCD(X=X, Y=all_Y, window_size=window, threshold=thres, delta=delta, eta=eta,
                                burn_in=burn_in, outlier_test=True)
                    drift_points, outlier_points = ipcd.fit()

                judge_size = 50  # Valid range: 50 samples before/after drift
                true_drift_points = [i for i in range(2000, len(X), 2000)]  # Simulated true drift points

                # Count TP, FP, FN
                TP = 0
                FP = 0
                FN = 0
                delay = []

                for detected in drift_points:
                    if any(((detected - actual) <= judge_size and (detected - actual) >= 0) for actual in
                           true_drift_points):
                        TP += 1
                    else:
                        FP += 1
                    for actual in true_drift_points:
                        if (detected - actual) <= judge_size and (detected - actual) >= 0:
                            delay.append(detected - actual)

                # Count missed drifts
                for actual in true_drift_points:
                    if not any(((detected - actual) <= judge_size and (detected - actual) >= 0) for detected in
                               drift_points):
                        FN += 1

                end_time = time.time()
                execution_time = end_time - start_time
                time_list.append(execution_time)

                # Calculate metrics
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                if TP != 0:
                    F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
                else:
                    F1_Score = 0

                results = {
                    "Model": name,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "Precision": Precision,
                    "Recall": Recall,
                    "F1_Score": F1_Score,
                    "time": execution_time,
                    "delay": delay
                }
                results_list.append(results)
                print(num_r, num_window, num)
                F1score[num_r, num_window, num] = F1_Score

            df_results = pd.DataFrame(results_list)
            # Create global parameter table
            global_params = {
                "Parameter": ["window_size", "delta", "threshold", "burn_in", "judge_size", "X_dim", "rate"],
                "Value": [window, delta, thres, burn_in, judge_size, X.shape[1], r]
            }
            df_global_params = pd.DataFrame(global_params)

            # Save to Excel
            with pd.ExcelWriter(
                    f"out_simu_drift_detection_results_summary_rate_{r:.3f}___{window:.2f}_{eta:.4f}.xlsx") as writer:
                df_global_params.to_excel(writer, sheet_name="Global Parameters", index=False)
                df_results.to_excel(writer, sheet_name="Model Results", index=False)

print("Results summary saved to drift_detection_results_summary.xlsx")