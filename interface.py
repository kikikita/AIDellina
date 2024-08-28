import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
from pereezd import make_full_predict
import sys
import threading


def redirector(inputStr):
    console.insert(tk.INSERT, inputStr)


class IORedirector(object):
    def __init__(self, text_area):
        self.text_area = text_area


class StdoutRedirector(IORedirector):
    def write(self, str):
        self.text_area.insert("end", str)
        self.text_area.see("end")


def run_prediction():
    osp = osp_entry.get()
    date_open = datetime.strptime(date_open_entry.get(), "%Y-%m-%d")
    date_start = datetime.strptime(
        date_start_entry.get() or "2014-01-01", "%Y-%m-%d")
    date_until = datetime.strptime(
        date_until_entry.get() or "2029-12-01", "%Y-%m-%d")
    is_mt_cf = mt_cf_var.get()
    is_forecast = forecast_var.get()

    def run_in_thread():
        make_full_predict(osp, date_open, date_start, date_until,
                          is_forecast, is_mt_cf)

    threading.Thread(target=run_in_thread).start()


root = tk.Tk()
root.geometry("500x550")
root.title("AIDellina: Переезды")

osp_frame = tk.Frame(root)
osp_frame.pack(pady=10)

osp_label = tk.Label(osp_frame, text="Наименование ОСП:")
osp_label.pack()
osp_entry = tk.Entry(osp_frame)
osp_entry.pack()

date_open_frame = tk.Frame(root)
date_open_frame.pack(pady=10)

date_open_label = tk.Label(date_open_frame, text="Дата открытия:")
date_open_label.pack()
date_open_entry = tk.Entry(date_open_frame)
date_open_entry.pack(padx=10)

date_start_frame = tk.Frame(root)
date_start_frame.pack(pady=10)

date_start_label = tk.Label(date_start_frame, text="Данные с:")
date_start_label.pack()
date_start_entry = tk.Entry(date_start_frame)
date_start_entry.insert(0, "2014-01-01")
date_start_entry.pack(padx=10)

date_until_frame = tk.Frame(root)
date_until_frame.pack(pady=10)

date_until_label = tk.Label(date_until_frame, text="Прогноз до:")
date_until_label.pack()
date_until_entry = tk.Entry(date_until_frame)
date_until_entry.insert(0, "2029-12-01")
date_until_entry.pack(padx=10)

mt_cf_var = tk.IntVar()
mt_cf_check = tk.Checkbutton(
    root, text="Применить кф нагрузки к МТ", variable=mt_cf_var)
mt_cf_check.pack(pady=10)

forecast_var = tk.IntVar()
forecast_check = tk.Checkbutton(
    root, text="Использовать прогноз СПП", variable=forecast_var)
forecast_check.pack(pady=10)

make_forecast_button = tk.Button(
    root, text="Построить прогноз", command=run_prediction, padx=50, pady=20)
make_forecast_button.pack(pady=20)

console = scrolledtext.ScrolledText(root, width=60, height=30)
console.pack(pady=10)

sys.stdout = StdoutRedirector(console)

root.mainloop()
