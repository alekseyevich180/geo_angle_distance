import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.signal import argrelextrema

# 1. データを準備する
# datファイルからデータを読み込む (第一列をx, 第三列をyとして使用)
data = np.loadtxt('ICOHP.dat')  # 同じフォルダにあるdatファイルを読み込む
X = data[:, 0].reshape(-1, 1)  # 第一列をXに
y = data[:, 2]  # 第三列をyに

# 1.1 基于孤立森林的异常值检测
clf_X = IsolationForest(contamination=0.05, n_estimators=200, max_samples=50, random_state=42)  # X的异常检测
clf_y = IsolationForest(contamination=0.2, n_estimators=200, max_samples=50, random_state=42)  # y的异常检测

outliers_X = clf_X.fit_predict(X)  # 检测X的异常值 (-1 表示异常值)
outliers_y = clf_y.fit_predict(y.reshape(-1, 1))  # 检测y的异常值 (-1 表示异常值)

# 只保留非异常值的点
filtered_X = X[(outliers_X == 1) & (outliers_y == 1)]
filtered_y = y[(outliers_X == 1) & (outliers_y == 1)]

# 异常值检测后可视化
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='gray', alpha=0.6, label='Original Data')
plt.scatter(filtered_X, filtered_y, color='blue', label='Filtered Data', marker='x')
plt.title('Isolation Forest: Anomaly Detection')
plt.xlabel('Surface-IrO (°)')
plt.ylabel('ICOHP (eV)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. ガウス過程回帰モデルの作成
# カーネルの定義 (RBFカーネルを使用)
kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=20.0, length_scale_bounds=(0.5, 100.0))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3)

# 3. モデルの学習
gpr.fit(filtered_X, filtered_y)

# 最適なパラメータを出力
print("Optimized kernel parameters:")
print(gpr.kernel_)

# 4. 推論用データを生成
X_pred = np.linspace(filtered_X.min(), filtered_X.max(), 1000).reshape(-1, 1)

# 5. 推論 (平均と標準偏差)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# 6. プロット
plt.figure(figsize=(8, 6))
plt.scatter(filtered_X, filtered_y, color='black', label='Filtered Data', alpha=0.6, marker='x')
plt.plot(X_pred, y_pred, color='blue', label='Mean', linewidth=2)
plt.fill_between(X_pred.ravel(),
                 y_pred - 1.96 * sigma,
                 y_pred + 1.96 * sigma,
                 color='lightblue', alpha=0.5, label='Confidence Interval')
plt.title('Gaussian Process Regression')
plt.xlabel('Surface-IrO (°)')
plt.ylabel('ICOHP (eV)')
plt.ylim(-2.6, -1.8)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 7. 局部极值点的提取
print("Local Maxima:")
local_maxima = argrelextrema(y_pred, np.greater)[0]
for idx in local_maxima:
    print(f"X: {X_pred[idx][0]:.2f}, Y: {y_pred[idx]:.2f}")

print("\nLocal Minima:")
local_minima = argrelextrema(y_pred, np.less)[0]
for idx in local_minima:
    print(f"X: {X_pred[idx][0]:.2f}, Y: {y_pred[idx]:.2f}")

# 8. 数据与预测值的比较
pred_y = gpr.predict(filtered_X)
plt.figure(figsize=(6, 6))
plt.scatter(filtered_y, pred_y, alpha=0.7, marker='x')
plt.plot([filtered_y.min(), filtered_y.max()],
         [filtered_y.min(), filtered_y.max()], ls="--", color="red", label="y = x")
plt.xlabel("True Values (eV)")
plt.ylabel("Predicted Values (eV)")
plt.title("True vs Predicted Values")
plt.legend()
plt.tight_layout()
plt.show()
