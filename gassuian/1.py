import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. データを準備する
# datファイルからデータを読み込む (第一列をx, 第二列をyとして使用)
data = np.loadtxt('ICOHP.dat')  # 同じフォルダにあるdatファイルを読み込む
X = data[:, 0].reshape(-1, 1)  # 第一列をXに
y = data[:, 2]  # 第二列をyに

# 1.1 各1区間内の方差をチェックして過大なデータを削除
bins = np.arange(np.floor(X.min()), np.ceil(X.max()) + 1, 1)  # 1区間ごとのビン
bin_indices = np.digitize(X.ravel(), bins)  # 各点のビンを特定
filtered_X = []
filtered_y = []
recommended_variance = 0.005  # 推奨方差の値をやや緩和

for i in range(1, len(bins)):
    bin_mask = bin_indices == i
    bin_X = X[bin_mask]
    bin_y = y[bin_mask]
    
    while np.var(bin_y) > recommended_variance and len(bin_y) > 4:  # 方差が大きい場合
        print(f"Before filtering: Variance = {np.var(bin_y):.4f}, Points = {len(bin_y)}")
        # 最も離れている複数の点を削除
        mean_y = np.mean(bin_y)
        distances = np.abs(bin_y - mean_y)
        remove_indices = np.argsort(distances)[-5:]  # 上位5つの点を削除
        bin_X = np.delete(bin_X, remove_indices, axis=0)
        bin_y = np.delete(bin_y, remove_indices)
        print(f"After filtering: Variance = {np.var(bin_y):.4f}, Points = {len(bin_y)}")
    
    filtered_X.extend(bin_X)
    filtered_y.extend(bin_y)

X = np.array(filtered_X).reshape(-1, 1)
y = np.array(filtered_y)

# 2. ガウス過程回帰モデルの作成
# カーネルの定義 (RBFカーネルを使用)
kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=20.0, length_scale_bounds=(0.5, 100.0))  # 長さスケールと定数値を調整
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3)  # alphaを増加

# 3. モデルの学習
gpr.fit(X, y)

# 最適なパラメータを出力
print("Optimized kernel parameters:")
print(gpr.kernel_)

# 4. 推論用データを生成
X_pred = np.linspace(min(X), max(X), 1000).reshape(-1, 1)  # 推論用データ点数をさらに増加

# 5. 推論 (平均と標準偏差)
y_pred, sigma = gpr.predict(X_pred, return_std=True)

# 6. プロット
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='black', label='Filtered Data', alpha=0.6, marker='x')
plt.plot(X_pred, y_pred, color='blue', label='Mean', linewidth=2)
plt.fill_between(X_pred.ravel(),
                 y_pred - 1.96 * sigma,
                 y_pred + 1.96 * sigma,
                 color='lightblue', alpha=0.5, label='Confidence Interval')
plt.title('Gaussian Process Regression')
plt.xlabel('Surface-IrO (°)')
plt.ylabel('ICOHP (eV)')
plt.ylim(-2, -1)  # 縦軸の範囲を制限
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 7. 結果を表示
print("Local Maxima:")
local_maxima = argrelextrema(y_pred, np.greater)[0]
for idx in local_maxima:
    print(f"X: {X_pred[idx][0]:.2f}, Y: {y_pred[idx]:.2f}")

print("\nLocal Minima:")
local_minima = argrelextrema(y_pred, np.less)[0]
for idx in local_minima:
    print(f"X: {X_pred[idx][0]:.2f}, Y: {y_pred[idx]:.2f}")

# 8. データと予測値の比較プロット
pred_y = gpr.predict(X)
plt.figure(figsize=(6, 6))
plt.scatter(y, pred_y, alpha=0.7, marker='x')
plt.plot([min(y), max(y)], [min(y), max(y)], ls="--", color="red", label="y = x")
plt.xlabel("True Values (eV)")
plt.ylabel("Predicted Values (eV)")
plt.title("True vs Predicted Values")
plt.legend()
plt.tight_layout()
plt.show()

# 9. 固定次元のプロット
# 例: 2次元データの場合、片方の次元を固定して予測値をプロット
if X.shape[1] == 1:  # データが1次元の場合はこの部分をスキップ
    print("Plotting fixed input visualization is not applicable for 1D data.")
else:
    fixed_inputs = 0  # 固定する次元のインデックス
    num_points = 100  # 可視化する点数
    fixed_value = np.mean(X[:, fixed_inputs])  # 固定する次元の平均値

    # 固定された次元を持つ新しいデータを作成
    grid_x = np.linspace(X[:, 1 - fixed_inputs].min(), X[:, 1 - fixed_inputs].max(), num_points)
    grid_fixed = np.full(grid_x.shape, fixed_value)

    if fixed_inputs == 0:
        test_data = np.column_stack([grid_fixed, grid_x])
    else:
        test_data = np.column_stack([grid_x, grid_fixed])

    pred_mean, pred_sigma = gpr.predict(test_data, return_std=True)

    # プロット
    plt.figure(figsize=(8, 6))
    plt.plot(grid_x, pred_mean, color='blue', label='Mean Prediction')
    plt.fill_between(grid_x, 
                     pred_mean - 1.96 * pred_sigma, 
                     pred_mean + 1.96 * pred_sigma, 
                     color='lightblue', alpha=0.5, label='Confidence Interval')
    plt.title('Fixed Input Visualization')
    plt.xlabel('Variable Dimension')
    plt.ylabel('Prediction')
    plt.legend()
    plt.tight_layout()
    plt.show()
