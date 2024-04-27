import pandas as pd
import numpy as np
import random
from scipy.optimize import fsolve
from prophet import Prophet


# def forecast(table_name: str, periods: int):
#     df = pd.read_excel(f'data/{table_name}.xlsx')
#     result = []
#     for i in range(1, len(df.columns)):
#         col_name = df.columns[i]
#         temp_df = df.iloc[:, [0, i]].rename({col_name: 'y'}, axis=1)
#         m = Prophet()
#         m.fit(temp_df)
#         future = m.make_future_dataframe(periods=periods, freq='MS')
#         forecast = m.predict(future)
#         m.plot(forecast, ylabel=col_name)
#         temp_df = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
#         temp_df.columns = ['ds', col_name, f'{col_name}_up',
#           f'{col_name}_low']
#         if i == 1:
#             result = temp_df
#         else:
#             result = result.join(temp_df.iloc[:, 1:4])
#     t_result = result.tail(periods).set_index('ds').T
#     t_result.to_excel(f'data/{table_name}_forecast.xlsx')
#     print('Прогноз построен')


def mt_coef_finder(cut_forecast: pd.DataFrame, weight: float):
    def func(x):
        return [(x[0] * np.sum(cut_forecast['mt_1'])) /
                (x[0] * np.sum(cut_forecast['mt_1']) +
                 np.sum(cut_forecast['mt_2'])) - weight]
    root = fsolve(func, 1)
    min_cf = 1 - (1 - root) / 2
    max_cf = 1 + (1 - root) / 2
    # if root < 1:
    mt_1_cf, mt_2_cf = min_cf, max_cf
    # else:
    #     mt_1_cf, mt_2_cf = max_cf, min_cf
    return mt_1_cf, mt_2_cf


def result_mt_changer(cut_df: pd.DataFrame, result: pd.DataFrame):
    cut_df['summ'] = cut_df['mt_1'] + cut_df['mt_2']
    weight = np.sum(cut_df['mt_1']) / np.sum(cut_df['summ'])
    year_list = result.index.year.unique()
    for i in year_list:
        mt_1_cf, mt_2_cf = mt_coef_finder(result.loc[f'{i}'], weight)
        result.loc[f'{i}', 'mt_1'] *= mt_1_cf[0]
        result.loc[f'{i}', 'mt_2'] *= mt_2_cf[0]
    return result


def tn_coef_finder(cut_df, periods):
    tn_coefs = {}
    for i in range(1, ((len(cut_df.columns)+1)//2)):
        cut_df[f'mt_tn_{i}'] = cut_df[f'mt_{i}']/cut_df[f'tn_{i}']
        # tn_coefs[f'tn_{i}'] = [f'mt_{i}', np.mean(cut_df[f'mt_tn_{i}'])]
        mean_cf = np.mean(cut_df[f'mt_tn_{i}'])
        tn_coefs[f'tn_{i}'] = [f'mt_{i}',
                               [random.uniform(mean_cf-2, mean_cf+2)
                                for _ in range(periods)]]
    return tn_coefs


def forecast(table_name: str, periods: int):
    df = pd.read_excel(f'data/{table_name}.xlsx')
    cut_df = df.tail(12).copy()
    result = []
    for i in range(1, len(df.columns)):
        col_name = df.columns[i]
        temp_df = df.iloc[:, [0, i]].rename({col_name: 'y'}, axis=1)
        m = Prophet()
        m.fit(temp_df)
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)
        # m.plot(forecast, ylabel=col_name)
        temp_df = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
        temp_df.columns = ['ds', col_name, f'{col_name}_upper',
                           f'{col_name}_lower']
        temp_df[col_name] = np.abs(temp_df[col_name])
        if i == 1:
            result = temp_df
        else:
            result = result.join(temp_df.iloc[:, 1:4])
    result = result.tail(periods).set_index('ds')
    result = result_mt_changer(cut_df, result)
    tn_coefs = tn_coef_finder(cut_df, periods)
    for k, v in tn_coefs.items():
        result[k] = result[v[0]] / v[1]
    t_result = result.T
    t_result.to_excel(f'data/{table_name}_forecast.xlsx')
    return t_result


table = str(input('Введите название таблицы: '))
periods = int(input('Укажите количество периодов : '))
forecast(table, periods)
