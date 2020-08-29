# data import
import pandas as pd
veriler = pd.read_csv('grup2.csv')
print(veriler)

# data discovery
eksikveri= veriler.isnull().sum()
print(eksikveri)

# data types view
types = veriler.dtypes
print(types)

# statistic tables
pd.set_option('display.width', 60, 'precision', 2)
istatistik = veriler.describe()
print('İstatistik tablosu: \n', istatistik)

# correlation matrix and grafics

import seaborn as sn
import matplotlib.pyplot as plt
pd.set_option('display.width', 60, 'precision', 2)
corrmatrix = veriler.corr(method='pearson')
print("Korelasyon matrix'i:\n", corrmatrix)
plt.figure(figsize=(10,7))
sn.heatmap(corrmatrix, annot=True)
plt.show()

# histogram chart

veriler.hist(figsize=(10,10), grid=False)
print("\nKolonların histogram grafiği:")
plt.show()

# Decomposition of x and y columns
x = veriler.iloc[:, 0:16]
y = veriler.iloc[:, 17:18]
print('\n', x.shape)
print('\n', y.shape, '\n')
# eğitim ve test verisinin ayrıştırılması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state=0)

# normalize process

from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(x)
x_norm = minMaxScaler.transform(x)
x_train_norm = minMaxScaler.transform(x_train)
x_test_norm = minMaxScaler.transform(x_test)

# we are modeling for all attributes
# First machine learning model is Linear Regression

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train_norm, y_train)

# regresion test and train part
y_reg_test_pred = reg.predict(x_test_norm)

# Test performance for first model Linear Regression r2 score

from sklearn.metrics import r2_score, mean_squared_error
reg_test_score = r2_score(y_test, y_reg_test_pred)
print("Regresyon modeli test performansı \nr2: ", reg_test_score)

# Train performance for first model Linear Regression predict
y_reg_train_pred = reg.predict(x_train_norm)

# Train performance for first model Linear Regression r2 score
reg_train_score = r2_score(y_train, y_reg_train_pred)

print("Regresyon modeli eğitim performansı \nr2: ", reg_train_score)
print('MSE eğitim : ', mean_squared_error(y_train, y_reg_train_pred))
print('MSE test : ', mean_squared_error(y_test, y_reg_test_pred))

# Second machine learning model Polinomial Regresion
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
poly= PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_norm)

# polinomial Regression test and train datasets aparting process
x_poly_train, x_poly_test, y_poly_train, y_poly_test = train_test_split(x_poly, y, test_size =0.3, random_state=0)

linreg2 = LinearRegression().fit(x_poly_train, y_poly_train)

# Polynomial model coefficients and success scores
print('polinom modelinin katsayıları (w): ', (linreg2.coef_))
print('Polinom Modelinin sabit (b) katsayısı: ', (linreg2.intercept_))
print('Polinom Modelinin R Kare Eğitim performansı: ', (linreg2.score(x_poly_train, y_poly_train)))
print('Polinom Modelinin R kare Test performansı: ', (linreg2.score(x_poly_test, y_poly_test)))
print('o olmayan öznitelik sayısı: ', np.sum(linreg2.coef_ !=0))

# y column test and train predict scores
pred_y_poly_train = linreg2.predict(x_poly_train)
pred_y_poly_test = linreg2.predict(x_poly_test)

# Polinomial model error rates
print('MSE eğitim : ', mean_squared_error(y_poly_train, pred_y_poly_train))
print('MSE test : ', mean_squared_error(y_poly_test, pred_y_poly_test))
print('Eğitim performansı (R kare): ', r2_score(y_poly_train, pred_y_poly_train))
print('Test performansı (R kare): ', r2_score(y_poly_test, pred_y_poly_test))

# Third machine learning model is decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0, max_depth=10)
tree.fit(x_train, y_train)

# Decision tree Predict scores
tree_pred_train = tree.predict(x_train)
tree_pred_test = tree.predict(x_test)

# model's performans scores
print('\n\n')
print('Eğitim performansı (R kare): ', r2_score(y_train, tree_pred_train))
print('Test performansı (R kare): ', r2_score(y_test, tree_pred_test))
print('Eğitim hata oranı (MSE): ', mean_squared_error(y_train, tree_pred_train))
print('Test hata oranı (MSE): ', mean_squared_error(y_test, tree_pred_test))

# now we doing same things after select the best features for our dataset
# Feature selection process
from sklearn.feature_selection import SelectKBest, f_regression

# useing the normalized data
test = SelectKBest(score_func=f_regression, k=10)
fit = test.fit(x_norm, y)

np.set_printoptions(precision=3)
cols = test.get_support(indices=True)
x_kbest = x_norm[:,cols]
print('\nözniteliklerin skorları: \n', fit.scores_[0:16])

# we set up to 3 model with selected new features
# we create test and train dataset with new features
x_traink, x_testk, y_traink, y_testk = train_test_split(x_kbest, y, test_size =0.3, random_state=0)

# Linear Regression
regk = LinearRegression()
regk.fit(x_traink, y_traink)

# Linear Regression Test predict
y_regk_test_pred = regk.predict(x_testk)

# Linear Regression test performance r2 score
regk_test_score = r2_score(y_testk, y_regk_test_pred)
print("Regresyon modeli test performansı \nr2: ", regk_test_score)

# Linear Regression train datasets predict
y_regk_train_pred = regk.predict(x_traink)

# Regression train performance r2 score
regk_train_score = r2_score(y_traink, y_regk_train_pred)
print("Regresyon modeli eğitim performansı \nr2: ", regk_train_score)
print('MSE eğitim : ', mean_squared_error(y_traink, y_regk_train_pred))
print('MSE test : ', mean_squared_error(y_testk, y_regk_test_pred))

# Polinomial Regression machine learning model
polyk= PolynomialFeatures(degree=2)
x_polyk = poly.fit_transform(x_kbest)

linreg3k = LinearRegression().fit(x_traink, y_traink)

# Polynomial model coefficients and success scores with new features
print('yeni özniteliklerle\n')
print('polinom modelinin katsayıları (w): ', (linreg3k.coef_))
print('Polinom Modelinin sabit (b) katsayısı: ', (linreg3k.intercept_))
print('Polinom Modelinin R Kare Eğitim performansı: ', (linreg3k.score(x_traink, y_traink)))
print('Polinom Modelinin R kare Test performansı: ', (linreg3k.score(x_testk, y_testk)))
print('o olmayan öznitelik sayısı: ', np.sum(linreg3k.coef_ !=0))

# y column test and train predict scores
y_polyk_train_pred = linreg3k.predict(x_traink)
y_polyk_test_pred = linreg3k.predict(x_testk)

# Model's error scores for test and train dataset
print('MSE eğitim : ', mean_squared_error(y_traink, y_polyk_train_pred))
print('MSE test : ', mean_squared_error(y_testk, y_polyk_test_pred))

# Third machine learning model is desicion tree (with new features)
from sklearn.tree import DecisionTreeRegressor
treek = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0, max_depth=10)
treek.fit(x_traink, y_traink)

# predict scores
y_treek_train_pred = treek.predict(x_traink)
y_treek_test_pred = treek.predict(x_testk)

# Performance scores (for train, test dataset r2 scores and mean squared error)
print('\nkarar ağaçları yeni özniteliklerle:\n')
print('Eğitim performansı (R kare): ', r2_score(y_traink, y_treek_train_pred))
print('Test performansı (R kare): ', r2_score(y_testk, y_treek_test_pred))
print('Eğitim hata oranı (MSE): ', mean_squared_error(y_traink, y_treek_train_pred))
print('Test hata oranı (MSE): ', mean_squared_error(y_testk, y_treek_test_pred))

# Parameter optimization process
# we doing paramater optimazation process for the all machine learning models
# Polinomial Regresion model Parameter optimization
print('\nparametre optimizasyonu\n')
print('Polinom Modelinin parametreleri: \n', polyk._get_param_names())
import time
print('Polinom Dereceleri\n')
for a in range(1,5):
    tic=time.time()
    polinom_opt_pred = PolynomialFeatures(degree=a)
    x_poly_opt = polinom_opt_pred.fit_transform(x_kbest)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_poly_opt, y, test_size =0.3, random_state=0)
    polimodelopt = LinearRegression()
    polimodelopt.fit(x_train1, y_train1)
    pred_poly_y_train1= polimodelopt.predict(x_train1)
    pred_poly_y_test1= polimodelopt.predict(x_test1)
    print('derece ', a, 'için :\n')
    print('MSE Eğitim: ', mean_squared_error(y_train1, pred_poly_y_train1))
    print('MSE Test: ', mean_squared_error(y_test1, pred_poly_y_test1))
    print('R Kare Eğitim Skoru: ', r2_score(y_train1, pred_poly_y_train1))
    print('R Kare Test Skoru: ', r2_score(y_test1, pred_poly_y_test1))
    toc= time.time()
    print(toc-tic, 'saniye Geçti\n')
    
# Regresion Decision Tree Parameter optimization
from sklearn.model_selection import GridSearchCV

param_grid = {
        'max_depth': [5, 10, 15, 20, 25],
        'min_impurity_decrease': [0, 0.001, 0.005, 0.01],
        'min_samples_split': [10, 20, 30, 40, 50]}

grid_model= GridSearchCV(tree, param_grid, cv=5  ,n_jobs=-1)
grid_model.fit(x_traink, y_traink)
best_para= grid_model.best_params_
print('regresyon karar ağaçları için en iyi parametreler: \n', best_para)

# Modeling with best paramaters
y_pred_test_opt= grid_model.predict(x_testk)
y_pred_train_opt= grid_model.predict(x_traink)
print('\nParametre optimizasyonu soonrası model performans ölçümü: \n')
print('MSE Eğitim: ', mean_squared_error(y_traink, y_pred_train_opt))
print('MSE Test: ', mean_squared_error(y_testk, y_pred_test_opt))
print('R Kare Eğitim Skoru: ', r2_score(y_traink, y_pred_train_opt))
print('R Kare Test Skoru: ', r2_score(y_testk, y_pred_test_opt))

# best parameters for linear model
parameters = {
        'fit_intercept':('True', 'False'),
        'normalize':('True', 'False'),
        'copy_X':('True', 'False')}

lin_grid_model= GridSearchCV(regk, parameters, cv=5  ,n_jobs=-1)
lin_grid_model.fit(x_traink, y_traink)
lin_best_para = lin_grid_model.best_params_
print('Doğrusal regresyon için en iyi parametreler: \n', lin_best_para)

# best parameter for linear model
y_lin_pred_opt_test = lin_grid_model.predict(x_testk)
y_lin_pred_opt_train = lin_grid_model.predict(x_traink)

# after doing parameter optimization performance scores(for test, train dataset eror score, r2 score )
print('\nParametre optimizasyonu sonrası model performans ölçümü: \n')
print('MSE Eğitim: ', mean_squared_error(y_traink, y_lin_pred_opt_train))
print('MSE Test: ', mean_squared_error(y_testk, y_lin_pred_opt_test))
print('R Kare Eğitim Skoru: ', r2_score(y_traink, y_lin_pred_opt_train))
print('R Kare Test Skoru: ', r2_score(y_testk, y_lin_pred_opt_test))





