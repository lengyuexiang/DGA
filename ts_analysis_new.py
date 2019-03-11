
import pandas as pd
import numpy as np
import os


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def load(file='/chentian2B.csv'):
    # load dataset
    path = os.getcwd()+'/data/'
    dataframe = pd.read_csv(path + file)
    dataframe = normalization(dataframe, 1)

    return dataframe

def normalization(t, type):
    sel_col = ['H2_volume','CO_volume','CO2_volume',
                 'CH4_volume','C2H4_volume','C2H6_volume','total_hydrocarbons']
    # normalize mean = 0; std = 1
    if type == 0:
        for col in sel_col:
            # t[col] = np.log10(t[col])
            t[col] = (t[col] - t[col].mean()) / (t[col].std())
    # normalize from 0 to 1
    else:
        for col in sel_col:
            # t[col] = np.log10(t[col])
            t[col] = (t[col] - t[col].min()) / (t[col].max() - t[col].min())

    return t


def select_column(dataframe,column='H2_volume',lag=20):
    # selected column
    col_n = ['time',column]
    H2 = pd.DataFrame(dataframe,columns=col_n)
    # convert string/object to data time format
    H2['time'] = pd.to_datetime(H2['time'],format='%Y-%m-%d')
    # sort the time ascending
    H2 = H2.sort_values(by=['time'],ascending=True)

    # plot the scatter points
    # plt.plot(H2['time'], H2['H2_volume'],'b.')
    # plt.show()

    # the dataset consisting of several years
    # reshape the data by 12 days
    # lag = 12                # lag
    length = H2.shape[0]    #
    splitframe = pd.DataFrame()
    x_col = []
    for i in range(lag,0,-1):
        splitframe['t-'+str(i)] = H2[column].shift(i)
        x_col.append('t-'+str(i))
        # splitframe['t-' + str(i)] = H2['time'].shift(i)
        splitframe['t'] = H2[column].values
    # print(splitframe.head(13))
    splitframe = splitframe[lag+1:]

    # plot_acf(splitframe[0])
    # plt.show()

    # split variables and response
    X = splitframe[x_col]
    X = X.values
    y = splitframe['t']
    y = y.values

    return X,y

def split(X,y,percent=0.8):

    train_len = int(X.shape[0] * percent)
    X_train,y_train,X_test,y_test = X[:train_len],y[:train_len], X[train_len:],y[train_len:]

    return X_train,y_train,X_test,y_test

def smape_metric(predict, actual):

    predict =np.array(predict)
    actual = np.array(actual)

    return sum(abs(predict - actual) / (abs(predict) + abs(actual)) / 2.0) / len(predict)

def base(X, y):
    y_hat = np.mean(X, axis=1)
    mae = mean_absolute_error(y, y_hat)
    print('base line mean value of input is: %.3f' % (mae))
    mae = round(mae, 4)

    smape = smape_metric(y_hat, y)
    print('base line mean value of input is: %.3f' % (smape))

    return mae, smape

def randomforest(X, y):
    mses_mean = []
    mses_std = []
    # fit random forest model
    estimators = [50, 100, 200, 300, 400, 500]
    for estimator in estimators:
        RF = RandomForestRegressor(n_estimators=estimator, random_state=1)

        MAE = -cross_val_score(RF, X, y, cv=10, scoring='neg_mean_absolute_error')
        mses_mean.append(np.mean(MAE))
        mses_std.append(np.std(MAE))
            # print MAE
    rf_mse = round(np.min(mses_mean), 4)
    rf_std = round(mses_std[mses_mean.index(min(mses_mean))], 4)
    print("Random forest average MSE is : ", rf_mse)
    print("Random forest std MSE is : ", rf_std)

def regression(X_train, y_train,X_test,y_test):

    smape_list = []
    mae_list = []
    # fit random forest model
    estimators = [50,100,200,300,400,500]
    for estimator in estimators:
        RF = RandomForestRegressor(n_estimators=estimator, random_state=1)
        RF.fit(X_train, y_train)
        prediction = RF.predict(X_test)
        smape = smape_metric(prediction, y_test)
        smape_list.append(smape)
        mae = mean_absolute_error(prediction,y_test)
        mae_list.append(mae)
    rf_mae = round(np.min(mae_list),4)
    rf_smape = round(np.min(smape_list), 4)
    print("Random forest smape is : ", rf_smape)
    print("Random forest mae is : ", rf_mae)

    # fit lasso model find best parameters
    smape_list = []
    mae_list = []
    alphas = [0.0001,0.001,0.01,0.1,1,10]
    for alpha in alphas:
        Lasso = linear_model.Lasso(alpha=alpha)
        Lasso.fit(X_train, y_train)
        prediction = Lasso.predict(X_test)
        smape = smape_metric(prediction, y_test)
        smape_list.append(smape)
        mae = mean_absolute_error(prediction, y_test)
        mae_list.append(mae)
    lasso_mae = round(np.min(mae_list), 4)
    lasso_smape = round(np.min(smape_list), 4)
    print("Lasso smape is : ", lasso_smape)
    print("Lasso mae is : ", lasso_mae)

    # # fit linear model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    prediction = lm.predict(X_test)
    smape = round(smape_metric(prediction, y_test),4)
    mae = round(mean_absolute_error(prediction, y_test),4)
    print("linear model smape is : ", round(smape,4))
    print("linear model mae is : ", round(mae,4))

    # #fit svm model
    C = [0.0001,0.001,0.01,0.1,1,10]
    smape_list = []
    mae_list = []
    for c in C:
        svr = SVR(C=c)
        svr.fit(X_train, y_train)
        prediction = svr.predict(X_test)
        smape = smape_metric(prediction, y_test)
        smape_list.append(smape)
        mae = mean_absolute_error(prediction, y_test)
        mae_list.append(mae)

    svr_mae = round(np.min(mae_list), 4)
    svr_smape = round(np.min(smape_list), 4)
    print("svm smape is : ", svr_smape)
    print("svm mae is : ", svr_mae)

    #fit nerual network model
    hidden_layer_sizes = [50,100,150,200,250,300]
    smape_list = []
    mae_list = []
    for size in hidden_layer_sizes:
        nn = MLPRegressor(hidden_layer_sizes=size)
        nn.fit(X_train, y_train)
        prediction = nn.predict(X_test)
        smape = smape_metric(prediction, y_test)
        smape_list.append(smape)
        mae = mean_absolute_error(prediction, y_test)
        mae_list.append(mae)
    nn_mae = round(np.min(mae_list), 4)
    nn_smape = round(np.min(smape_list), 4)
    print("nerual network smape is : ", nn_smape)
    print("nerual network mae is : ", nn_mae)

    #
    # # fit decision tree model
    smape_list = []
    mae_list = []
    depths = [2,3,4,5,6,7]
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X_train,y_train)
        prediction = tree.predict(X_test)
        smape = smape_metric(prediction, y_test)
        smape_list.append(smape)
        mae = mean_absolute_error(prediction, y_test)
        mae_list.append(mae)
    tree_mae = round(np.min(mae_list), 4)
    tree_smape = round(np.min(smape_list), 4)
    print("nerual network smape is : ", tree_smape)
    print("nerual network mae is : ", tree_mae)
    #     if metric == 'mse':
    #         MSE = -cross_val_score(tree, X, y, cv=10, scoring='neg_mean_squared_error')
    #         mses_mean.append(np.mean(MSE))
    #         mses_std.append(np.std(MSE))
    #     # print MSE
    #     elif metric == 'mae':
    #         MAE = -cross_val_score(tree, X, y, cv=10, scoring='neg_mean_absolute_error')
    #         mses_mean.append(np.mean(MAE))
    #         mses_std.append(np.std(MAE))
    #     # print MAE
    # print("tree model average MSE is : ", np.min(mses_mean))
    # print("tree model std MSE is : ", np.min(mses_std))
    # tree_mse = round(np.min(mses_mean),4)
    # tree_std = round(mses_std[mses_mean.index(min(mses_mean))],4)
    #
    # # fit quadratic regression
    # quadratic = PolynomialFeatures(degree=2)
    # X_quad = quadratic.fit_transform(X)
    # if metric == 'mse':
    #     MSE = -cross_val_score(lm, X_quad, y, cv=10, scoring='neg_mean_squared_error')
    #     print("quadratic regression model average MSE is : ", np.mean(MSE))
    #     print("quadratic regression model std MSE is : ", np.std(MSE))
    #     poly_mse = round(np.mean(MSE),4)
    #     poly_std = round(np.std(MSE),4)
    # elif metric == 'mae':
    #     MAE = -cross_val_score(lm, X_quad, y, cv=10, scoring='neg_mean_absolute_error')
    # # print MAE
    #     print("quadratic regression model average MSE is : ", np.mean(MAE))
    #     print("quadratic regression model std MSE is : ", np.std(MAE))
    #     poly_mse = round(np.mean(MAE),4)
    #     poly_std = round(np.std(MAE),4)
    #
    #
    # mses = str([rf_mse,lasso_mse,lm_mses,svr_mse,nn_mse,tree_mse,poly_mse])
    # stds = str([rf_std,lasso_std,lm_std,svr_std,nn_std,tree_std,poly_std])

    maes = [rf_mae,lasso_mae,mae,svr_mae,nn_mae,tree_mae]
    smapes = [rf_smape,lasso_smape,smape,svr_smape,nn_smape,tree_smape]
    return maes, smapes

def xx_regression(file, f, column):
    dataframe = load(file)
    X, y = select_column(dataframe, column)
    X_train, y_train, X_test, y_test = split(X, y, percent=0.8)
    maes, smapes = regression(X_train, y_train, X_test, y_test)
    # f.write(str(maes) + '\n')
    # f.write(str(smapes) + '\n')
    return maes,smapes

def all_feature_regression(file,f):
    # file = 'xiamen1.csv'
    columns = ['H2_volume','CO_volume','CO2_volume','CH4_volume','C2H4_volume','C2H6_volume','total_hydrocarbons']
    # columns = ['H2_volume', 'CO_volume']
    maes_list = []
    smapes_list = []
    for column in columns:
        # f.write(column + '\n')
        maes, smapes = xx_regression(file ,f, column = column)
        maes_list.append(maes)
        smapes_list.append(smapes)

    maes_list = np.array(maes_list)
    smapes_list = np.array(smapes_list)

    return maes_list, smapes_list


def all_data_regression():
    log = 'regression_new.log'
    f = open(log, 'w')
    path = os.getcwd() + '/data'
    maes_array = []
    smapes_array = []
    for file in os.listdir(path):
        if file.endswith('csv'):
            print (file)
            # f.write(file+'\n')
            maes_list, smapes_list = all_feature_regression(file, f)
            maes_array.append(maes_list)
            smapes_array.append(smapes_list)

    maes_array = np.array(maes_array)
    smapes_array = np.array(smapes_array)

    maes_avg = np.mean(maes_array, axis=0)
    smapes_avg = np.mean(smapes_array, axis=0)
    f.write(str(maes_avg)+'\n')
    f.write(str(smapes_avg)+'\n')

    f.close()

if __name__ == '__main__':
    # main_program()
    all_data_regression()