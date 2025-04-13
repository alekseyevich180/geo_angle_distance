import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 异常值检测函数
def remove_outliers(X, y, method='zscore', z_threshold=2, iqr_multiplier=1, mad_threshold=2):
    from scipy.stats import zscore
    if method == 'zscore':
        y_zscores = zscore(y)
        mask = np.abs(y_zscores) < z_threshold

    elif method == 'iqr':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask = (y >= lower_bound) & (y <= upper_bound)

    elif method == 'mad':
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        mask = np.abs(y - median_y) / mad_y < mad_threshold

    else:
        raise ValueError("Invalid method. Choose 'zscore', 'iqr', or 'mad'.")

    return X[mask], y[mask]

# 数据准备
data = np.loadtxt('Ir2.dat')
X = data[:, 0].reshape(-1, 1)
y = -data[:, 2]

# 异常值检测
X_filtered, y_filtered = remove_outliers(X, y, method='zscore')

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_filtered)
y_scaled = scaler_y.fit_transform(y_filtered.reshape(-1, 1)).ravel()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 决策树回归模型
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# 训练集和测试集 R² 分数
train_r2 = r2_score(y_train, tree.predict(X_train))
test_r2 = r2_score(y_test, tree.predict(X_test))

# 预测结果
X_pred = np.linspace(X_scaled.min(), X_scaled.max(), 1000).reshape(-1, 1)
y_pred = tree.predict(X_pred)

# 将预测结果转换回原始坐标
X_pred_original = scaler_X.inverse_transform(X_pred).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

# 可视化结果
plt.figure(figsize=(4, 4))
plt.scatter(scaler_X.inverse_transform(X_scaled), scaler_y.inverse_transform(y_scaled.reshape(-1, 1)), 
            color='black', alpha=0.6, label='Filtered Data', marker='x')
plt.plot(X_pred_original, y_pred_original, color='red',  linewidth=2) #label='Mean Prediction',
plt.title(f'Decision Tree Regression \nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}', fontsize=16)
plt.xlabel('O-Ir-O angle (°)', fontsize=12)
plt.ylabel('-IpCOHP (eV)', fontsize=12)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()

# 真值 vs 预测值 (训练集)
y_train_pred_original = scaler_y.inverse_transform(tree.predict(X_train).reshape(-1, 1)).ravel()

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

# 计算残差并可视化分布
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 6))
plt.hist(residuals_train, bins=30, alpha=0.6, color='blue', label='Train Residuals')
plt.hist(residuals_test, bins=30, alpha=0.6, color='orange', label='Test Residuals')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
