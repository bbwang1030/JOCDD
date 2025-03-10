import numpy as np
from numpy.linalg import inv, norm
from scipy.stats import norm as stats_norm
import matplotlib.pyplot as plt
from page_hinkley import PageHinkley  # 引用已经定义的 PageHinkley 类
from sklearn.linear_model import LinearRegression
from statsmodels.robust.robust_linear_model import RLM
import math
np.random.seed(4)
# IPOD 方法的函数实现
def IPODFUN(X, Y, H, sigma, betaInit=None, method="hard", TOL=1e-4):
    N = len(Y)
    gamma = np.zeros(N)  # 保持 gamma 为一维数组
    theta = sigma * np.sqrt(2 * np.log(N))

    if betaInit is None:
        r = Y.flatten()  # 确保 r 是一维数组
        if method == "hard":
            # 使用一维索引来更新 gamma
            gamma[np.abs(r) > theta] = r[np.abs(r) > theta]
        elif method == "soft":
            gamma[r > theta] = r[r > theta] - theta
            gamma[r < -theta] = r[r < -theta] + theta
    else:
        gamma_old = gamma
        r0 = Y.flatten() - np.dot(H, Y).flatten()  # 确保 r0 和 r 一维
        niter = 0
        theta = sigma * np.sqrt(2 * np.log(N)) * np.sqrt(1 - np.diag(H))
        while True:
            niter += 1
            if niter == 1:
                r = Y.flatten() - np.dot(X, betaInit)  # r 一维
            else:
                r = np.dot(H, gamma_old).flatten() + r0  # r 一维

            if method == "hard":
                gamma = np.zeros(N)  # 保持 gamma 为一维数组
                gamma[np.abs(r) > theta] = r[np.abs(r) > theta]  # 使用一维索引
            elif method == "soft":
                gamma = np.zeros(N)
                gamma[r > theta] = r[r > theta] - theta[r > theta]
                gamma[r < -theta] = r[r < -theta] + theta[r < -theta]

            if np.max(np.abs(gamma - gamma_old)) < TOL:
                break
            else:
                gamma_old = gamma
    return {"gamma": gamma, "ress": r}


# IPOD_new 函数
def IPOD_new(X, Y, H, method="hard", TOL=1e-4, eta=1/500):
    if X is None:
        r = 0
    else:
        r = X.shape[1]

    N = len(Y)
    ress = []
    gammas = []

    if X is None:  # X is None
        betaInit = None
        step=math.ceil((norm(Y, ord=np.inf) + 1)*eta)
        lambdas = np.arange(round(norm(Y, ord=np.inf) + 1), 0, -step)
    else:
        # Robust linear model estimation
        model = RLM(Y, X).fit()
        betaInit = model.params
        tmp = np.dot((np.eye(H.shape[0]) - H).T, Y) / np.sqrt(1 - np.diag(H))
        # print(round(norm(tmp, ord=np.inf) + 1))
        step = math.ceil((norm(tmp, ord=np.inf) + 1) *eta)
        # print(step)
        lambdas = np.arange(round(norm(tmp, ord=np.inf) + 1), 0, -step)

    # 限制 lambdas 的数量，避免过多计算
    # lambdas = lambdas[:1000]
    # print(333,len(lambdas))
    for sigma in lambdas:
        sigma = sigma / np.sqrt(2 * np.log(N))
        result = IPODFUN(X, Y, H, sigma, betaInit, method=method, TOL=TOL)
        gammas.append(result["gamma"])
        ress.append(result["ress"])
    # print(444)
    gammas = np.column_stack(gammas)
    ress = np.column_stack(ress)

    DF = np.sum(np.abs(gammas) > 1e-5, axis=0)
    # print(gammas,DF)
    if X is not None:
        Q, _ = np.linalg.qr(X, mode='complete')
        X_unscaled = Q[:, r:].T
        Y_eff = np.dot(X_unscaled, Y.flatten())
        # print(Y,Y.flatten())
        # 计算 sigmaSqEsts
        # 计算分子部分
        numerator = np.sum((Y_eff[:, np.newaxis] - np.dot(X_unscaled, gammas)) ** 2, axis=0)
        # 计算分母部分
        denominators = len(Y_eff) - DF
        # print(denominators)
        # 将 denomiators 转换为 NumPy 数组
        denominators = np.array(denominators)

        # 找到 denomiators 中为 0 的索引
        zero_indices = np.where(denominators == 0)[0]

        # 将 sigmaSqEsts 中对应于 denomiators 中为 0 的值的元素赋值为无穷大
        sigmaSqEsts = np.divide(numerator, denominators, out=np.full_like(numerator, np.inf), where=(denominators != 0))

    else:
        sigmaSqEsts = np.sum((Y[:, np.newaxis] - gammas) ** 2, axis=0) / (len(Y) - DF)

    sigmaSqEsts[sigmaSqEsts < 0] = 0
    # valid_indices = DF <= N - r
    # if not np.any(valid_indices):
    #     raise ValueError("No valid DF values found after filtering.")
    #
    # gammas = gammas[:, valid_indices]
    # sigmaSqEsts = sigmaSqEsts[valid_indices]
    # ress = ress[:, valid_indices]
    # DF = DF[valid_indices]
    # Check if any column in gamma corresponds to DF greater than N - r
    if not np.all(DF <= N - r):
        valid_indices = DF <= N - r
        gammas = gammas[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

        # print(DF)
        # print(1111111111111)

    # Identify redundant columns in gammas
    redundant_inds = np.where(np.sum(np.abs(gammas[:, 1:] - gammas[:, :-1]), axis=0) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Identify redundant sigma square estimates
    redundant_inds = np.where(np.abs(sigmaSqEsts[1:] - sigmaSqEsts[:-1]) < 0.001)[0] + 1
    if len(redundant_inds) > 0:
        gammas = np.delete(gammas, redundant_inds, axis=1)
        lambdas = np.delete(lambdas, redundant_inds)
        sigmaSqEsts = np.delete(sigmaSqEsts, redundant_inds)
        ress = np.delete(ress, redundant_inds, axis=1)
        DF = np.delete(DF, redundant_inds)

    # Check if any column in gamma corresponds to DF greater than N/2
    if not np.all(DF <= N / 2):
        valid_indices = DF <= N / 2
        gammas = gammas[:, valid_indices]
        sigmaSqEsts = sigmaSqEsts[valid_indices]
        lambdas = lambdas[valid_indices]
        ress = ress[:, valid_indices]
        DF = DF[valid_indices]

    # print(DF)
    riskEst = ((N - r) * np.log(sigmaSqEsts * (N - r - DF) / (N -
                                                           r)) + (np.log(N - r) + 1) * (DF + 1)) / (N - r)

    optSet = np.where(riskEst == np.min(riskEst))[0]
    gammaOptInd = optSet[np.where(DF[optSet] == np.min(DF[optSet]))[0][0]]
    gammaOpt = np.array(gammas)[:, gammaOptInd]
    resOpt = np.array(ress)[:, gammaOptInd]
    tau = np.median(np.array(ress)[np.array(gammas)[:, gammaOptInd] == 0, gammaOptInd])
    # 计算 p 值
    # tau = np.median(np.abs(ress), axis=0)  # 使用残差的中位数作为尺度估计
    resOpt_scale = ress / tau  # 计算标准化残差
    p = 2 * stats_norm.sf(np.abs(resOpt_scale))  # 计算双侧 p 值

    return {
        "gamma": gammaOpt,
        "sigmaSqEsts": sigmaSqEsts,
        "ress": resOpt,
        "DF": DF,
        "p": p  # 返回 p 值
    }

class IPCD:
    def __init__(self, X, Y, window_size, threshold, delta, eta, burn_in=25, outlier_test = True):
        self.X = X
        self.Y = Y
        self.window_size = window_size
        self.threshold = threshold
        self.delta = delta
        self.eta = eta
        self.burn_in = burn_in
        self.outlier_test = outlier_test
        self.drift_points = []
        self.outlier_points = []
        self.outlier_warning = "not_outlier_warning"
        self.outlier_warning_before = False
        self.outlier_check_before = False
        self.ph_detector = PageHinkley(delta=delta, threshold=threshold, burn_in=burn_in, direction="positive")

    def fit(self):
        n = len(self.Y)
        # reference_window_X = self.X[:self.window_size]
        # reference_window_Y = self.Y[:self.window_size]
        # reference_H = np.dot(np.dot(reference_window_X, inv(np.dot(reference_window_X.T, reference_window_X))),
        #                      reference_window_X.T)
        # reference_result = IPOD_new(reference_window_X, reference_window_Y, reference_H)
        # gamma = reference_result["gamma"].reshape((self.window_size, 1))
        # beta_ols = np.dot(
        #     inv(np.dot(reference_window_X.T, reference_window_X)),
        #     np.dot(reference_window_X.T, reference_window_Y - gamma),
        # )
        # mean = np.mean(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)
        # std = np.std(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)

        res = [0]
        for i in range(self.window_size, n):
            test_window_X = self.X[i - self.window_size :i]
            test_window_Y = self.Y[i - self.window_size :i]

            new_X = self.X[i- self.window_size]
            new_Y = self.Y[i- self.window_size]
            # print(1)
            if self.ph_detector.drift_state == "drift" or i==self.window_size:
                reference_window_X = test_window_X
                reference_window_Y = test_window_Y
                reference_H = np.dot(
                    np.dot(reference_window_X, inv(np.dot(reference_window_X.T, reference_window_X))),
                    reference_window_X.T
                )
                # print(111)
                reference_result = IPOD_new(reference_window_X, reference_window_Y, reference_H, eta=self.eta)
                # print(222)
                gamma = reference_result["gamma"].reshape((self.window_size, 1))

                self.ph_detector.reset()

                beta_ols = np.dot(
                    inv(np.dot(reference_window_X.T, reference_window_X)),
                    np.dot(reference_window_X.T, reference_window_Y - gamma),
                )

                mean = np.mean(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)
                std = np.std(reference_window_Y - np.dot(reference_window_X, beta_ols) - gamma)
                # print(mean,std)
            residual = abs(new_Y - np.dot(new_X, beta_ols))

            res.append(float(residual))

            if self.outlier_test:
                outlier_warning = abs(new_Y - np.dot(new_X, beta_ols) - mean) > (3 * std)
                outlier_check = abs(new_Y - np.dot(new_X, beta_ols) - mean) > (6 * std)

                # if self.outlier_check_before and not outlier_check and self.drift_state_before != "drift":
                if self.outlier_warning_before and not outlier_warning and self.drift_state_before != "drift":
                    self.outlier_warning = "outlier_warning"
                    if self.outlier_check_before:
                        self.outlier_warning = "outlier"
                        self.outlier_points.append(i - self.window_size-1)
                        # print(i - self.window_size-1)
                    # if (i - self.window_size-1)==58006 or (i - self.window_size-1)==58008:
                    #     print(res[-2])
                else:
                    self.outlier_warning = "not_outlier_warning"
                    self.ph_detector.update(np.array([res[-2]]))
                # print(f"PH statistic at index {i}: {self.ph_detector._page_hinkley_differences[-1]},{self.ph_detector._theta_threshold[-1]}")
                self.outlier_warning_before = outlier_warning
                self.outlier_check_before = outlier_check
                self.drift_state_before = self.ph_detector.drift_state


                if self.ph_detector.drift_state == "drift":
                    self.drift_points.append(i- self.window_size-1)


        return self.drift_points, self.outlier_points