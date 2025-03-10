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
window_size = 1200
threshold = 80# The threshold is 1% of the window size, default is 20
threshold_outlier = 10
delta = 0.01
burn_in = 20
d=10
bins = 50
v = np.random.choice([False,True], size=20000)
w = np.random.choice([False,True], size=20000)

x = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.1,0.7,0.1,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125], size=20000)
z = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.1,0.7,0.1,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125], size=20000)

r1 = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.1,0.7,0.1,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125], size=20000)
r2 = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.1,0.7,0.1,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.0125], size=20000)

x[2000:3000] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=1000)
z[2000:3000] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=1000)

# Drift 1
x[5000:7500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)
z[5000:7500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)


#Drift 2
r1[10000:12500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)
r2[10000:12500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)


#Drift 1
x[15000:17500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)
z[15000:17500] = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], p=[0.0125,0.0125,0.0125,0.0125,0.0125,0.0125,0.1,0.7,0.1,0.0125,0.0125], size=2500)

condition1 = v & w
condition2 = v | w
condition3 = z < 0.5 + 0.3 * np.sin(3*np.pi*x)

#y = np.where(condition1 | (condition2 & condition3),np.ones(20000, dtype=np.int8), np.zeros(20000, dtype=np.int8))
y= condition1*1 + condition2*1 + condition3*1

data = pd.DataFrame([v,w,x,z,r1,r2,y]).transpose()
data.columns = ['x1','x2','x3','x4','x5','x6','y']
data['x1'] = data['x1'].astype('int')
data['x2'] = data['x2'].astype('int')
data['y'] = data['y'].astype('int32')


X = data[['x1','x2','x3','x4','x5','x6']].values
# Y = rolling_mean_filled.values
Y=data['y'].values
all_Y = Y.reshape(-1,1)
X = X.astype(np.float64)
all_Y = all_Y.astype(np.float64)

n=20000
rate=0.1
eta=1/2
for outlier in range(int(n * rate)):
    # Generate mean and standard deviation for outliers
    mean = np.random.uniform(low=3, high=5, size=1)  # Generate random mean
    std_dev = np.random.uniform(low=0, high=1, size=1)  # Generate random standard deviation
    random_number = np.random.normal(mean, std_dev, size=1)  # Generate outlier value

    # 50% probability of generating an outlier with a negative mean
    if np.random.rand() < 0.5:
        random_number = np.random.normal((-mean), std_dev, size=1)

    # Add outlier at a specific position (e.g., following the outlier definition rule)
    # Assume outliers are added to all_Y at a specific position, e.g., at index 400*(outlier+1)
    outlier_index = int(n / (bins * (int(n * rate / bins) + 1)) * (outlier + 1))
    if outlier_index < len(all_Y):  # Ensure the index exists
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
true_drift_points = [2000,3000,5000,7500,10000,12500,15000,17500]
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

    judge_size = 100  # Valid range is 100 points before and after drift
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
with pd.ExcelWriter(f"new{rate:.2f}out_mixed_drift_detection_results_summary_rate_.xlsx") as writer:
    df_global_params.to_excel(writer, sheet_name="Global Parameters", index=False)
    df_results.to_excel(writer, sheet_name="Model Results", index=False)

print("Results summary saved to drift_detection_results_summary.xlsx")