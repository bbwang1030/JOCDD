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
# Data generation

np.random.seed(2)
window_size = 1000
threshold =30 # The threshold is 1% of the window size, default is 20
threshold_outlier = 10
delta = 0.01
burn_in = 20
d=10
bins=50

x1 = np.random.uniform(0, 1, 20000)
x2 = np.random.uniform(0, 1, 20000)
x3 = np.random.uniform(0, 1, 20000)
x4 = np.random.uniform(0, 1, 20000)
x5 = np.random.uniform(0, 1, 20000)
x6 = np.random.uniform(0, 1, 20000)
x7 = np.random.uniform(0, 1, 20000)
x8 = np.random.uniform(0, 1, 20000)
x9 = np.random.uniform(0, 1, 20000)
x10 = np.random.uniform(0, 1, 20000)

data = pd.DataFrame([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).transpose()
data.columns = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']

data['x1'][5000:10000] = np.random.uniform(-1, 2, 5000)
data['x3'][5000:10000] = np.random.uniform(-1, 2, 5000)
data['x5'][5000:10000] = np.random.uniform(-1, 2, 5000)
data['y_org'] = 10 * np.sin(np.pi * data['x1'] * data['x2']) + 20 * (data['x3'] - 0.5) ** 2 + 10 * data['x4'] + 5 * \
                data['x5'] + np.random.normal(0, 1)
data['y_1'] = 10 * np.sin(np.pi * data['x4'] * data['x5']) + 20 * (data['x2'] - 0.5) ** 2 + 10 * data['x1'] + 5 * data[
    'x3'] + np.random.normal(0, 1)
data['y_2'] = 10 * np.sin(np.pi * data['x2'] * data['x5']) + 20 * (data['x4'] - 0.5) ** 2 + 10 * data['x3'] + 5 * data[
    'x1'] + np.random.normal(0, 1)
data['y_global_gradual'] = data['y_org']

prob = 0.05
for i in range(5001, 6000):
    p = np.random.binomial(1, prob)

    if p > 0:
        data['y_global_gradual'][i] = data['y_1'][i]

    if (i % 100) == 0:
        prob += 0.1

data['y_global_gradual'][6000:7500] = data['y_1'][6000:7500]

prob = 0.05
for i in range(7501, 8500):
    p = np.random.binomial(1, prob)

    if p > 0:
        data['y_global_gradual'][i] = data['y_2'][i]

    if (i % 100) == 0:
        prob += 0.1

data['y_global_gradual'][8500:] = data['y_2'][8500:]
prob = 0.05
for i in range(15001, 16000):
    p = np.random.binomial(1, prob)

    if p > 0:
        data['y_global_gradual'][i] = data['y_1'][i]

    if (i % 100) == 0:
        prob += 0.1

data['y_global_gradual'][16000:17500] = data['y_1'][6000:7500]

prob = 0.05
for i in range(17501, 18500):
    p = np.random.binomial(1, prob)

    if p > 0:
        data['y_global_gradual'][i] = data['y_2'][i]

    if (i % 100) == 0:
        prob += 0.1

data['y_global_gradual'][18500:] = data['y_2'][18500:]
X = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']].values
Y=data['y_global_gradual'].values
all_Y = Y.reshape(-1,1)
X = X.astype(np.float64)
all_Y = all_Y.astype(np.float64)

n=20000
rate=0.05
eta=1/8
for outlier in range(int(n * rate)):
    # Generate mean and standard deviation for outliers
    mean = np.random.uniform(low=3, high=5, size=1)   # Generate random mean
    std_dev = np.random.uniform(low=0, high=1, size=1)  # Generate random standard deviation
    random_number = np.random.normal(mean, std_dev, size=1)  # Generate outlier value

    # 50% probability of generating an outlier with a negative mean
    if np.random.rand() < 0.5:
        random_number = np.random.normal((-mean), std_dev, size=1)

    # Add outlier at a specific position (e.g., following the outlier definition rule)
    # Assume outliers are added to all_Y at a specific position, e.g., at index 400*(outlier+1)
    outlier_index = int(n / (bins * (int(n * rate / bins) + 1)) * (outlier + 1))
    if outlier_index < len(all_Y):   # Ensure the index exists
        all_Y[int(outlier_index)] += random_number[0]
        print(f"Added outlier at index {outlier_index}: {random_number[0]}")
    else:
        print(f"Index {outlier_index} out of bounds.")

# Define a general drift detection function
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

# Model dictionary
models = {
    "Lasso": Lasso(alpha=0.01, max_iter=1000),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "Bagging": BaggingRegressor(random_state=42),
    "OSELM": OSELMRegressor(n_hidden=100),
    "GradientBoosting":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Linear":LinearRegression(),
    "Ipod":IPCD(X=X, Y=all_Y, window_size=window_size,threshold=threshold,delta=delta,eta=eta,burn_in=burn_in)
}

# Simulated true drift points
true_drift_points = [5000,6000,7500,8500,10000,16000,17500]
# print(true_drift_points)
results_list = []

# Iterate through each model
for name, model in models.items():
    print(f"Processing model: {name}")
    start_time = time.time()
    if name !="Ipod":
        drift_points = drift_detection_with_ph(model, X, all_Y, window_size, delta, threshold, burn_in)
    elif name =="Ipod":
        ipcd = IPCD(X=X, Y=all_Y ,window_size=window_size, threshold=threshold, delta=delta, eta=eta,burn_in=burn_in, outlier_test = True)
        drift_points, outlier_points = ipcd.fit()
    end_time = time.time()
    execution_time = end_time - start_time # Compute execution time

    judge_size = 100 # Valid range is 100 points before and after drift
    TP = 0
    FP = 0
    FN = 0

    for detected in drift_points:
        # Check if detected point is within the valid range of an actual drift point
        if any(abs(detected - actual) <= judge_size for actual in true_drift_points):
            TP += 1 # True Positive
        else:
            FP += 1  # False Positive


    # Count missed drift points
    for actual in true_drift_points:
        if not any(abs(detected - actual) <= judge_size for detected in drift_points):
            FN += 1  # False Negative

    # Compute Precision and Recall
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_Score = 2 * (Precision * Recall)/(Precision+Recall)if (Precision+Recall) > 0 else 0
    # false_negative = FN / len(true_drift_points) if len(drift_points) > 0 else 0


    results = {
        "Model": name,
        "TP":TP,
        "FP":FP,
        "FN":FN,
        "Precision": Precision,
        "Recall": Recall,
        "F1_Score":F1_Score,
        "Execution Time (s)": execution_time
        # "False Negatives": false_negative,
    }
    results_list.append(results)



# Save results to Excel
df_results = pd.DataFrame(results_list)
# Create global parameter table
global_params = {
    "Parameter": ["window_size", "delta", "threshold", "burn_in", "judge_size", "X_dim", "rate","eta"],
    "Value": [window_size, delta, threshold, burn_in, judge_size, X.shape[1], rate,eta]
}
df_global_params = pd.DataFrame(global_params)

# Save to Excel
with pd.ExcelWriter(f"new{rate:.2f}out_fri_drift_detection_results_summary_rate_.xlsx") as writer:
    df_global_params.to_excel(writer, sheet_name="Global Parameters", index=False)
    df_results.to_excel(writer, sheet_name="Model Results", index=False)

print("Results summary saved to drift_detection_results_summary.xlsx")
