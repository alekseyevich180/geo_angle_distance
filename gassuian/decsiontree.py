import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 加载 iris 数据集
iris_data = load_iris(as_frame=True)
iris_df = iris_data.frame

# 选择特征 sepal_length 和目标变量 petal_length
X = iris_df[['sepal length (cm)']]
Y = iris_df[['petal length (cm)']]

# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 决策树深度范围
depths = range(1, 11)

# 存储 R^2 的值
train_r2_scores = []
test_r2_scores = []

# 不同深度下训练回归树并计算 R^2
for depth in depths:
    tree_reg_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg_model.fit(X_train, Y_train)
    Y_pred_train = tree_reg_model.predict(X_train)
    Y_pred_test = tree_reg_model.predict(X_test)
    train_r2_scores.append(r2_score(Y_train, Y_pred_train))
    test_r2_scores.append(r2_score(Y_test, Y_pred_test))

results = pd.DataFrame({
    'Depth': depths,
    'Train R^2': train_r2_scores,
    'Test R^2': test_r2_scores
})


results.to_excel("decision_tree_r2_scores.xlsx", index=False)


plt.figure(figsize=(8, 6))
plt.plot(depths, train_r2_scores, label='Train R^2', marker='o')
plt.plot(depths, test_r2_scores, label='Test R^2', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('R^2 Score')
plt.title('Decision Tree Depth vs R^2 Scores')
plt.legend()
plt.grid()
plt.show()

# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# データのロード
iris = load_iris(as_frame=True)
iris_data = iris.frame

# 特徴量と目標値の選択
X = iris_data[['sepal length (cm)']]
Y = iris_data[['petal length (cm)']]

# training dataとtest dataに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 决定木の深さを1-10まで変化させる
depths = range(1, 11)
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []

for depth in depths:
    # 回帰木のモデル作成
    tree_reg_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_reg_model.fit(X_train, Y_train)

    # train dataとtest dataを使ってY値を予測
    Y_pred_train = tree_reg_model.predict(X_train)
    Y_pred_test = tree_reg_model.predict(X_test)

    # 決定係数 (R^2) の計算
    train_r2_scores.append(r2_score(Y_train, Y_pred_train))
    test_r2_scores.append(r2_score(Y_test, Y_pred_test))

    # RMSE の計算
    train_rmse_scores.append(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
    test_rmse_scores.append(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))

# 結果をDataFrameに保存
results = pd.DataFrame({
    'Depth': depths,
    'Train R^2': train_r2_scores,
    'Test R^2': test_r2_scores,
    'Train RMSE': train_rmse_scores,
    'Test RMSE': test_rmse_scores
})

# RMSE と R^2 の変化をプロット
plt.figure(figsize=(12, 6))

# R^2 のプロット
plt.subplot(1, 2, 1)
plt.plot(depths, train_r2_scores, label='Train R^2', marker='o')
plt.plot(depths, test_r2_scores, label='Test R^2', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('R^2 Score')
plt.title('R^2 Scores vs Tree Depth')
plt.legend()
plt.grid()

# RMSE のプロット
plt.subplot(1, 2, 2)
plt.plot(depths, train_rmse_scores, label='Train RMSE', marker='o')
plt.plot(depths, test_rmse_scores, label='Test RMSE', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('RMSE')
plt.title('RMSE vs Tree Depth')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 結果を表示
results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

iris_data = sns.load_dataset('iris')
iris_data.info()

iris_data.head()

X = iris_data[['sepal_length']].values
Y = iris_data['petal_width'].values

#training dataとtest dataに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

models = {
    "GLM (Linear Regression)": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=1),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=3, random_state=1)
}

results = []

for name, model in models.items():

    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    # R^2とRMSE
    train_r2 = r2_score(Y_train, Y_pred_train)
    test_r2 = r2_score(Y_test, Y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(Y_train, Y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_test))

    results.append({
        "Model": name,
        "Train R^2": train_r2,
        "Test R^2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })
    # 残差プロット
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_pred_train, Y_pred_train - Y_train, c='blue', marker='o', label='Train', alpha=0.7)
    plt.scatter(Y_pred_test, Y_pred_test - Y_test, c='red', marker='o', label='Test', alpha=0.7)
    plt.axhline(y=0, color='black', lw=2)
    plt.xlabel("Pred")
    plt.ylabel("Residual")
    plt.title(f"{name} Residuals")
    plt.legend()
    plt.show()


results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8, 6))
for name, model in models.items():
    Y_pred_test = model.predict(X_test)
    plt.scatter(Y_test, Y_pred_test, label=name, alpha=0.7)
plt.plot(np.linspace(0, 3, 100), np.linspace(0, 3, 100), 'r', label="y = x")
plt.xlim([0, 10])
plt.ylim([-10, 10])
plt.legend()
plt.tight_layout()
plt.show()