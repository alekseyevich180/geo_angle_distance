import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from scipy.stats import zscore

# 1. 数据准备
# 从文件读取数据 (第一列为 X，第三列为 y)
data = np.loadtxt('ICOHP.dat')  # 请确保 'ICOHP.dat' 文件路径正确
X = data[:, 0].reshape(-1, 1)  # 第一列为 X
y = data[:, 2]  # 第三列为 y

# 数据质量检查（剔除异常值）
y_zscores = zscore(y)
X = X[np.abs(y_zscores) < 3]
y = y[np.abs(y_zscores) < 3]

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 2. 核函数定义
# RBF 核函数
kernel_rbf = C(1.0, (1e-3, 1e5)) * RBF(length_scale=10.0, length_scale_bounds=(1e-4, 1e3))
# Matern 核函数
kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1))

# 选择核函数
kernel = kernel_matern

# 3. 高斯过程回归模型
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-1)

# 4. 数据分割与交叉验证
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 模型训练
gpr.fit(X_train, y_train)

# 交叉验证评分
scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring="r2")
print(f"Cross-validation R^2 scores: {scores}")
print(f"Mean R^2 score: {scores.mean():.3f}")

# 5. 模型预测
X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# 反标准化预测值
X_pred_original = scaler_X.inverse_transform(X_pred).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
sigma_original = scaler_y.scale_ * sigma

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(
    scaler_X.inverse_transform(X_scaled),
    scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
    color='black', alpha=0.6, label='Filtered Data'
)
plt.plot(X_pred_original, y_pred_original, color='blue', label='Mean Prediction', linewidth=2)
plt.fill_between(
    X_pred_original,
    y_pred_original - 1.96 * sigma_original,
    y_pred_original + 1.96 * sigma_original,
    color='lightblue', alpha=0.5, label='Confidence Interval'
)
plt.title('Gaussian Process Regression (Without Box-Cox)')
plt.scatter(filtered_X, filtered_y, color='blue', label='Filtered Data', marker='x')
plt.xlabel('O-Ir-O (°)')
plt.ylabel('ICOHP (eV)')
plt.legend()
plt.tight_layout()
plt.show()

# 6. 超参数优化
param_grid = {
    "kernel": [
        C(1.0) * RBF(length_scale=1.0),
        C(1.0) * Matern(length_scale=1.0, nu=1.5)
    ],
    "alpha": [1e-2, 1e-3, 1e-4]
}
gpr_grid = GaussianProcessRegressor()
grid_search = GridSearchCV(gpr_grid, param_grid, cv=5, scoring="r2")
grid_search.fit(X_scaled, y_scaled)

print("Best parameters:", grid_search.best_params_)

# 7. 真值 vs 预测值
y_train_pred = gpr.predict(X_train)
y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()

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

# 8. 残差分布
residuals = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel() - y_train_pred_original
plt.figure(figsize=(8, 4))
plt.scatter(scaler_X.inverse_transform(X_train), residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Residuals Distribution")
plt.xlabel("X (Original Scale)")
plt.ylabel("Residuals (eV)")
plt.show()
