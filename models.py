import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def model(df, model_type, parameter_criterion,parameter_cv, split_size):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    st.markdown('**1.2. 데이터 크기**')
    st.write('데이터 개수 / 피쳐 개수')
    st.info(X.shape)

    st.markdown('**1.3. 상관관계 **')
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    st.pyplot(fig)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=split_size/100)

    st.subheader('2. 예측 결과')
    st.write('데이터를 학습 중 입니다...')

    if model_type == 'Random forest':

        param_grid = {
            'max_depth': [10, 20, 50],
            'n_estimators': [200, 400, 1000]
        }

        clf = RandomForestRegressor(criterion=parameter_criterion)

    elif model_type == 'custom':
        pass

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=parameter_cv, verbose=2)
    grid_search.fit(X_train, Y_train)
    best_grid = grid_search.best_estimator_

    st.markdown('**2.1. 학습 데이터**')
    Y_pred_train = best_grid.predict(X_train)
    cols = df.columns.values[:-1]

    st.write('Permutation importances (train set)')
    result = permutation_importance(best_grid, X_train, Y_train, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=cols[sorted_idx])
    fig.tight_layout()
    st.pyplot(fig)

    st.write('결정 계수 ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('학습 RMSE:')
    st.info(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
    #st.write('Best hyperparameters:')
    #st.info(best_grid)


    st.markdown('**2.2. 시험 데이터**')
    Y_pred_test = best_grid.predict(X_test)

    st.write('Permutation importances (test set)')
    result = permutation_importance(best_grid, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=cols[sorted_idx])
    fig.tight_layout()
    st.pyplot(fig)

    st.write('결정 계수($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('테스트 RMSE:')
    st.info(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))

    st.write('예측값과 실제값의 관계 분포:')
    xx = np.linspace(min(Y_test),max(Y_test),100)
    fig = plt.figure(figsize=(3, 3))
    plt.plot(Y_test, Y_pred_test, '.')
    plt.plot(xx, xx,'--')
    plt.xlabel('$y$')
    plt.ylabel('$\at{y}$')
    st.pyplot(fig)