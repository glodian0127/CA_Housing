import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import lasso_path, enet_path
from yellowbrick.regressor import AlphaSelection
from yellowbrick.regressor import ManualAlphaSelection
from yellowbrick.regressor.alphas import manual_alphas
from itertools import cycle
import sys
import io
import seaborn as sns
import matplotlib


BEST_ALPHA = 850.5


def get_data(path):
    raw_data = pd.read_csv(path, header=0)
    return raw_data


def get_data_head(data):
    data_head = open('data_head_output.txt', 'w')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(data.head(), file=data_head)
    data_head.close()


def get_data_info(data):
    buffer = io.StringIO()
    data.info(verbose=True, buf=buffer)
    s = buffer.getvalue()
    with open('data_info_output.txt', 'w', encoding='utf-8') as f:
        f.write(s)


def get_data_description(data):
    data_description = open('data_description_output.txt', 'w')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(data.describe(), file=data_description)
    data_description.close()


def category_data_type(data):
    category_type = open('category_type.txt', 'w')
    print(data['ocean_proximity'].value_counts(), file=category_type)
    category_type.close()


def data_visualization(data):
    data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1, color='orange')
    plt.savefig('density.png')
    plt.close()


def get_data_correlation(data):
    sns.set(context='paper', font='monospace')
    corr_matrix = data.corr()
    fig, axe = plt.subplots(figsize=(20, 15))
    color_pattern = sns.diverging_palette(275, 240, center='light', as_cmap=True)
    sns.heatmap(corr_matrix, vmax=1, square=True, cmap=color_pattern, annot=True)
    plt.savefig('data_correlation.png')
    plt.close()


def category_convert(data):
    dum_df = pd.get_dummies(data.ocean_proximity)
    data_new = pd.concat([data, dum_df], axis=1)
    data_new = data_new.drop(['ocean_proximity'], axis=1)
    return data_new


def data_fill_nan(data):
    new_data = data.fillna(0)
    return new_data


def scale_data(data):
    data_x = data.iloc[:, :-6]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data_x.to_numpy())
    columns_name = list(data_x.columns)
    df_scaled = pd.DataFrame(df_scaled, columns=columns_name)
    data_scaled = pd.concat([df_scaled, data.iloc[:, -5:]], axis=1)
    data_scaled = pd.concat([data_scaled, data.iloc[:, -6]], axis=1)
    return data_scaled


def cv_curve_lasso(data):
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    alphas = np.arange(500, 1600, 0.5)
    model = LassoCV(alphas=alphas, cv=5, random_state=19)
    vis = AlphaSelection(model)
    vis.fit(data_x, data_y)
    vis.show(outpath='cv_curve_lasso.png')


def parameter_coefficient(data):
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    lasso_model = Lasso(alpha=BEST_ALPHA)
    lasso_model.fit(data_x, data_y)
    coef_df = pd.Series(lasso_model.coef_, index=data_x.columns)

    para_coefs = open('parameter_coefs.txt', 'w')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(coef_df, file=para_coefs)
    para_coefs.close()


def get_lasso_path(data):
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    eps = 5e-3
    alphas_lasso, coefs_lasso, _ = lasso_path(data_x, data_y, eps=eps, fit_intercept=False)

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    plt.title('Lasso_Path')
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.axis('tight')
    plt.savefig('Lasso_Path.png')
    plt.close()


def elastic_net_selection(data):
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    cv_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
                            normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, copy_X=True, verbose=0,
                            n_jobs=-1, positive=False, random_state=19, selection='cyclic')
    cv_model.fit(data_x, data_y)
    opt_alpha = cv_model.alpha_
    opt_l1_ratio = cv_model.l1_ratio_
    opt_iter = cv_model.n_iter_

    elastic_net = open('elastic_net_output.txt', 'w')
    print('Best Alpha: %.8f'%opt_alpha, '.\n', 'Best L1_ratio: %.3f'%opt_l1_ratio, '.\n', '# Iteration: %d'%opt_iter,
          file=elastic_net)
    elastic_net.close()


if __name__ == "__main__":
    data = get_data('data/Kaggle_housing-CA.csv')
    get_data_head(data)
    get_data_info(data)
    get_data_description(data)
    get_data_correlation(data)
    category_data_type(data)
    data_visualization(data)
    data_new = category_convert(data)
    data_filled = data_fill_nan(data_new)
    data_scaled = scale_data(data_filled)
    parameter_coefficient(data_scaled)
    get_lasso_path(data_scaled)
    elastic_net_selection(data_scaled)
    cv_curve_lasso(data_scaled)
    print(data_filled.info())
