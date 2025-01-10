import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import zscore
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, DotProduct, ConstantKernel as C

# 1. データを準備する
# datファイルからデータを読み込む (第一列をx, 第三列をyとして使用)
data = np.loadtxt('Ru.dat')  # 同じフォルダにあるdatファイルを読み込む
X = data[:, 0].reshape(-1, 1)  # 第一列をXに
y = data[:, 2]  # 第三列をyに

# 数据质量检查（剔除异常值）

y_zscores = zscore(y)
X = X[np.abs(y_zscores) < 3]
y = y[np.abs(y_zscores) < 3]


# 1.1 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 2. 核函数优化 (尝试不同核函数)
# RBF 核函数
#kernel_rbf = C(1.0, (1e-3, 1e5)) * RBF(length_scale=10.0, length_scale_bounds=(1e-4, 1e3))

# Matern 核函数 (常用于非平滑问题)
#kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=20.0, nu=1.5) 
#kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=10.0, length_scale_bounds=(1e-4, 1e3), nu=1.5)
#kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1))
# 选择最优的核函数
#kernel = kernel_matern
kernels = {
    "RBF": C(1.0) * RBF(length_scale=10.0),
    "RBF2" : C(1.5, (1e-1, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-5, 1e3)),
    "Matern": C(1.0) * Matern(length_scale=10.0, nu=1.5),
    "kernel_matern" : C(1.2, (1e-3, 1e4)) * Matern(length_scale=20.0, nu=1.5), 
    "kernel_matern2" : C(1.0, (1e-1, 1e2)) * Matern(length_scale=10.0, length_scale_bounds=(1e-5, 1e3), nu=1.5),
    "Matern + WhiteKernel" : C(1.0, (1e-3, 1e5)) * Matern(length_scale=12.0, nu=1.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e1)),
    "Matern + WhiteKernel2": C(1.0) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-6)
}
# 3. ガウス過程回帰モデルの作成
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_Rumizer=15, alpha=1e-2)
for name, kernel in kernels.items():
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gpr.fit(X_train, y_train)
    
    log_likelihood = gpr.log_marginal_likelihood_value_
    num_params = gpr.kernel_.n_dims
    aic = 2 * num_params - 2 * log_likelihood
    
    print(f"Kernel: {name}, AIC: {aic:.3f}")
#gpr = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", max_iter_predict=1000, alpha=1e-2)

# 4. 数据分割与交叉验证

scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring="r2")
print(f"Cross-validation R^2 scores: {scores}")
print(f"Mean R^2 score: {scores.mean():.3f}")

# 7️⃣ 可视化预测结果
X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# 反标准化
X_pred_original = scaler_X.inverse_transform(X_pred).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
sigma_original = scaler_y.scale_ * sigma

# 可视化预测结果
plt.figure(figsize=(10, 4))
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled.reshape(-1, 1)), color='black', alpha=0.6, label='Filtered Data')
plt.plot(X_pred_original, y_pred_original, color='blue', label='Mean Prediction', linewidth=2)
plt.fill_between(X_pred_original, y_pred_original - 1.96 * sigma_original, y_pred_original + 1.96 * sigma_original, color='lightblue', alpha=0.5, label='Confidence Interval')
plt.title('Gaussian Process Regression with AIC Analysis')
plt.xlabel('O-Ru-O (°)')
plt.ylabel('ICOHP (eV)')
#plt.ylim(-4.8, -4.3)
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

residuals = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel() - y_train_pred_original
plt.figure(figsize=(8, 4))
plt.scatter(scaler_X.inverse_transform(X_train), residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Residuals Distribution")
plt.xlabel("X (Original Scale)")
plt.ylabel("Residuals (eV)")
plt.show()