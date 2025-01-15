import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import zscore
from sklearn.metrics import r2_score
from scipy.optimize import fmin_l_bfgs_b
import warnings
from sklearn.exceptions import ConvergenceWarning


# 异常值检测函数
# method: 'zscore', 'iqr', or 'mad'
def remove_outliers(X, y, method='zscore', z_threshold=2, iqr_multiplier=1, mad_threshold=2):
    if method == 'zscore':
        # Z-score 方法
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < z_threshold

    elif method == 'iqr':
        # IQR 方法
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)

    elif method == 'mad':
        # MAD 方法
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        mask = np.abs(y - median_y) / mad_y < mad_threshold

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]

# 1. データを准备する
data = np.loadtxt('Ir.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 2]

# 比较三种异常值检测方法
detection_methods = ['zscore', 'iqr', 'mad']
results = {}

for method in detection_methods:
    # 异常值检测
    X_filtered, y_filtered = remove_outliers(X, y, method=method)

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_filtered)
    y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 核函数优化
    kernels = {
        "RBF": C(1.0) * RBF(length_scale=10.0),
        "RBF2" : C(1.5, (1e-1, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-6, 1e3)),
        "Matern": C(1.0) * Matern(length_scale=10.0, nu=1.5),
        "kernel_matern" : C(1.2, (1e-3, 1e4)) * Matern(length_scale=20.0, nu=1.5), 
        "kernel_matern2" : C(1.0, (1e-1, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-6, 1e3), nu=1.5),
        "Matern + WhiteKernel" : C(1.0, (1e-3, 1e5)) * Matern(length_scale=12.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1)),
        "Matern + WhiteKernel2": C(1.0) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-6),
        "kernel": C(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-6, 1e2)) + WhiteKernel(noise_level=1e-6)
    }

    method_results = []

    for name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                alpha=1e-2,
                
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gpr.fit(X_train, y_train)  
       
       
        gpr.fit(X_train, y_train)

        log_likelihood = gpr.log_marginal_likelihood_value_
        num_params = gpr.kernel_.n_dims
        aic = 2 * num_params - 2 * log_likelihood

        # 使用交叉验证计算 R² 分数
        scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring='r2')
        mean_r2 = scores.mean()

        method_results.append((name, aic, mean_r2))

    # 保存每种方法的所有核函数的 AIC 和 R² 值
    results[method] = {
        'method_results': method_results,
        'data_points': len(y_filtered)
    }

# 输出比较结果并选择最佳方法和核函数
best_method = None
best_kernel = None
best_aic = float('inf')
best_r2 = -float('inf')

for method, result in results.items():
    print(f"Method: {method}")
    for kernel_name, aic_value, r2_value in result['method_results']:
        print(f"  Kernel: {kernel_name}, AIC: {aic_value:.3f}, Mean R²: {r2_value:.3f}")
        if aic_value < best_aic and r2_value > best_r2:
            best_aic = aic_value
            best_r2 = r2_value
            best_method = method
            best_kernel = kernel_name
    print(f"  Remaining Data Points: {result['data_points']}")
    print("----------------------------------------")

# 选择最佳方法重新训练模型并可视化预测结果
print(f"Best Method: {best_method}, Best Kernel: {best_kernel}, Best AIC: {best_aic:.3f}, Best Mean R²: {best_r2:.3f}")
X_filtered, y_filtered = remove_outliers(X, y, method=best_method)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_filtered)
y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 使用最佳核函数重新训练
kernel = kernels[best_kernel]
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-2, optimizer="fmin_l_bfgs_b")
gpr.fit(X_train, y_train)

# 可视化预测结果
X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

X_pred_original = scaler_X.inverse_transform(X_pred).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
sigma_original = scaler_y.scale_ * sigma

train_r2 = r2_score(y_train, gpr.predict(X_train))
test_r2 = r2_score(y_test, gpr.predict(X_test))

plt.figure(figsize=(10, 4))
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled.reshape(-1, 1)), color='black', alpha=0.6, label='Filtered Data')
plt.plot(X_pred_original, y_pred_original, color='blue', label='Mean Prediction', linewidth=2)
plt.fill_between(X_pred_original, y_pred_original - 2.58 * sigma_original, y_pred_original + 2.58 * sigma_original, color='lightblue', alpha=0.5, label='Confidence Interval')
plt.title(f'Gaussian Process Regression with {best_method.capitalize()} Method and {best_kernel} Kernel\nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}')
plt.xlabel('O-Ir-O (°)')
plt.ylabel('ICOHP (eV)')
plt.legend()
plt.tight_layout()
plt.show()

# 7. 真值 vs 预测值
# 真值 vs 预测值 (训练集)
y_train_pred = gpr.predict(X_train)
y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()  # 修复：reshape(-1, 1)

# 可视化
plt.figure(figsize=(6, 6))
plt.scatter(
    scaler_y.inverse_transform(y_train.reshape(-1, 1)),
    y_train_pred_original,
    alpha=0.7,
    marker='x',
    label="Training Data"
)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    ls="--",
    color="red",
    label="y = x"
)
plt.xlabel("True Values (eV)")
plt.ylabel("Predicted Values (eV)")
plt.title("True vs Predicted Values (Training)")
plt.legend()
plt.tight_layout()
plt.show()

y_train_pred = gpr.predict(X_train)
y_test_pred = gpr.predict(X_test)
# Calculate residuals for train and test sets
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Plot residuals distribution (histogram and density plot)
plt.figure(figsize=(12, 6))

# Histogram for residuals
plt.hist(residuals_train, bins=30, alpha=0.6, color='blue', label='Train Residuals')
plt.hist(residuals_test, bins=30, alpha=0.6, color='orange', label='Test Residuals')

# Vertical line for zero residuals
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)

# Title and labels
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
