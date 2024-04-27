import pandas as pd
from prophet import Prophet


def forecast(table_name: str, periods: int):
    df = pd.read_excel(f'data/{table_name}.xlsx')
    result = []
    for i in range(1, len(df.columns)):
        col_name = df.columns[i]
        temp_df = df.iloc[:, [0, i]].rename({col_name: 'y'}, axis=1)
        m = Prophet()
        m.fit(temp_df)
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)
        m.plot(forecast, ylabel=col_name)
        temp_df = forecast[['ds', 'yhat']]
        temp_df.columns = ['ds', col_name]
        # temp_df = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
        # temp_df.columns = ['ds', col_name, f'{col_name}_up',
        # f'{col_name}_low']
        if i == 1:
            result = temp_df
        else:
            result = result.join(temp_df.iloc[:, 1:2])
    t_result = result.tail(periods).set_index('ds').T
    t_result.to_excel(f'data/{table_name}_forecast.xlsx')
    print('Прогноз построен')


table = str(input('Введите название таблицы: '))
periods = int(input('Укажите количество периодов : '))
forecast(table, periods)
