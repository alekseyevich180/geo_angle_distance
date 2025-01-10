from sklearn.model_selection import train_test_split
tree_reg_model = DecisionTreeRegressor
depths = range(1,10)
tree_reg_model.fit(X_train, Y_train)

Y_pred_train = tree_reg_model.predict(X_train)
Y_pred_test = tree_reg_model.predict(X_test)


from sklearn.metrics import root_mean_squared_error 
from sklearn.metrics import r2_score 

# 平均平方二乗誤差(RMSE)
print('RMSE 学習: %.2f, テスト: %.2f' % (
root_mean_squared_error(Y_train, Y_pred_train), # 学習
root_mean_squared_error(Y_test, Y_pred_test) # テスト
))

# 決定係数(R^2)
print('R^2 学習: %.2f, テスト: %.2f' % (
r2_score(Y_train, Y_pred_train), # 学習
r2_score(Y_test, Y_pred_test) # テスト
))



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 加载 iris 数据集
iris_data = load_iris(as_frame=True)
iris_df = iris_data.frame

# 选择特征 sepal_length 和目标变量 petal_length
X = iris_df[['sepal length (cm)']]
Y = iris_df[['petal length (cm)']]

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

plt.plot(depths, train_r2_scores, label='Train R^2', marker='o')
plt.plot(depths, test_r2_scores, label='Test R^2', marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('R^2 Score')
plt.title('Decision Tree Depth vs R^2 Scores')
plt.legend()
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestRegressor

models = {
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=3, random_state=1)
}