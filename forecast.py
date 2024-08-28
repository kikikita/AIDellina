import pandas as pd
import numpy as np
import random
from scipy.optimize import fsolve
from prophet import Prophet
# from darts import TimeSeries
# from darts.models import Prophet, ExponentialSmoothing
# from darts.dataprocessing.transformers import Scaler
# from darts.metrics import mape, r2_score

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
        temp_df.columns = ['ds', col_name, f'{col_name}_up',
                           f'{col_name}_low']
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


# def forecast(table_name: str, periods: int):
#     df = pd.read_excel(f'data/{table_name}.xlsx')
#     """
#     Функция для прогнозирования.

#     Параметры:
#     df (pd.DataFrame): исходный DataFrame.
#     periods (int): количество периодов для прогноза.

#     Возвращает:
#     result: DataFrame с прогнозными значениями.
#     """
#     # Read the data
#     cut_df = df.tail(6).copy()

#     # Prepare for metrics calculation
#     last_6_months = df.iloc[-6:]
#     model_dict = {}
#     mape_dict = {}
#     r2_dict = {}

#     result = []
#     for i in range(1, len(df.columns)):
#         col_name = df.columns[i]

#         temp_df = df[['ds', col_name]].copy()
#         temp_df.columns = ['ds', 'y']

#         scaler = Scaler()
#         series = scaler.fit_transform(
#             TimeSeries.from_dataframe(temp_df, 'ds', 'y'))
#         train_series = scaler.transform(
#             TimeSeries.from_dataframe(temp_df.iloc[:-6], 'ds', 'y'))

#         # models = [ExponentialSmoothing(), AutoARIMA(), Prophet()]
#         models = [Prophet(), ExponentialSmoothing()]
#         best_model = None
#         best_mape = float('inf')

#         # Iterate over all combinations of parameters
#         for model in models:
#             model.fit(train_series)
#             future = model.predict(6)

#             # Rescale the predicted series to the original scale
#             future = scaler.inverse_transform(future)

#             predicted = future.values()[-6:]
#             actual = last_6_months[col_name].values
#             predicted_ts = TimeSeries.from_values(predicted)
#             actual_ts = TimeSeries.from_values(actual)
#             current_mape = mape(actual_ts, predicted_ts)
#             current_r2 = r2_score(actual_ts, predicted_ts)

#             if current_mape < best_mape:
#                 best_mape = current_mape
#                 best_r2 = current_r2
#                 best_model = model

#         best_model.fit(series)
#         future = best_model.predict(int(periods))
#         future = scaler.inverse_transform(future)

#         temp_df = future.pd_dataframe().reset_index()
#         temp_df.columns = ['ds', col_name]

#         if i == 1:
#             result = temp_df
#         else:
#             result = result.join(temp_df.set_index('ds'), on='ds')

#         model_dict[col_name] = type(best_model).__name__
#         mape_dict[col_name] = best_mape
#         r2_dict[col_name] = best_r2

#     result = result.tail(periods).set_index('ds')
#     # result = result_mt_changer(cut_df, result)
#     tn_coefs = tn_coef_finder(cut_df, periods)
#     for k, v in tn_coefs.items():
#         result[k] = result[v[0]] / v[1]

#     df_r2 = pd.DataFrame({'R2': r2_dict})
#     df_mape = pd.DataFrame({'MAPE': mape_dict})
#     df_model = pd.DataFrame({'Model name': model_dict})
#     metrics = pd.concat([df_r2, df_mape, df_model], axis=1)
#     t_result = result.T
#     t_result.to_excel(f'data/{table_name}_forecast.xlsx')
#     print(metrics)


table = str(input('Введите название таблицы: '))
periods = int(input('Укажите количество периодов : '))
forecast(table, periods)
