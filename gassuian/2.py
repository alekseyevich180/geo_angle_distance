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
data = np.loadtxt('ICOHP.dat')  # 同じフォルダにあるdatファイルを読み込む
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

# 2. 核函数优化 (尝试不同核函数)
# RBF 核函数
kernel_rbf = C(1.0, (1e-3, 1e5)) * RBF(length_scale=10.0, length_scale_bounds=(1e-4, 1e3))

# Matern 核函数 (常用于非平滑问题)
#kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=20.0, nu=1.5)
#kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=10.0, length_scale_bounds=(1e-4, 1e3), nu=1.5)
kernel_matern = C(1.0, (1e-3, 1e5)) * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=1e-2)

# 选择最优的核函数
kernel = kernel_matern
# 3. ガウス過程回帰モデルの作成
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=1e-2)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-1)
#gpr = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", max_iter_predict=1000, alpha=1e-2)

# 4. 数据分割与交叉验证
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 训练模型
gpr.fit(X_train, y_train)

# 交叉验证评分
scores = cross_val_score(gpr, X_scaled, y_scaled, cv=5, scoring="r2")
print(f"Cross-validation R^2 scores: {scores}")
print(f"Mean R^2 score: {scores.mean():.3f}")

# 5. 模型预测
X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)  # 推论用数据点
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# 反标准化预测值
X_pred_original = scaler_X.inverse_transform(X_pred)  # 此处进行反标准化
X_pred_original = X_pred_original.ravel()  # 转为一维数组
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()  # 预测值转为一维
sigma_original = scaler_y.scale_ * sigma  # 标准化到原尺度的置信区间

# 打印数组形状检查
print("X_pred_original shape:", X_pred_original.shape)
print("y_pred_original shape:", y_pred_original.shape)
print("sigma_original shape:", sigma_original.shape)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(
    scaler_X.inverse_transform(X_scaled),
    scaler_y.inverse_transform(y_scaled.reshape(-1, 1)),
    color='black', alpha=0.6, label='Filtered Data'
)
plt.plot(X_pred_original, y_pred_original, color='blue', label='Mean Prediction', linewidth=2)
plt.fill_between(
    X_pred_original.ravel(),
    y_pred_original - 1.96 * sigma_original,
    y_pred_original + 1.96 * sigma_original,
    color='lightblue', alpha=0.5, label='Confidence Interval'
)
plt.title('Gaussian Process Regression (Optimized)')
plt.xlabel('Surface-IrO (°)')
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

