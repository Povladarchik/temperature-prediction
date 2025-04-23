import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU
from keras.src.legacy.saving import legacy_h5_format


# Глобальные переменные
model = None
scaler = None
time_series_df = None
months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']


# Функция для загрузки данных
def load_data():
    global time_series_df, scaler
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return None
    try:
        df = pd.read_csv(file_path, index_col=None) # Здесь читается txt файл
        df = df[(df['Год'] >= 2014) & (df['Год'] < 2025)] # Диапазон лет по условию
        data_list = []
        for _, row in df.iterrows():
            for i, month in enumerate(months):
                data_list.append([row['Год'] + i / 12, row[month]]) # Преобразуем данные по месяцам во временной ряд
        time_series_df = pd.DataFrame(data_list, columns=['Дата', 'Значение'])
        scaler = MinMaxScaler()
        time_series_df['Значение'] = scaler.fit_transform(time_series_df[['Значение']])
        messagebox.showinfo("Успех", "Данные успешно загружены!")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {e}")


# Функция для создания последовательностей
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


# Функция для обучения модели
def train_model():
    global model, time_series_df, scaler
    if time_series_df is None:
        messagebox.showerror("Ошибка", "Сначала загрузите данные.")
        return None
    try:
        status_label.config(text="Обучение начато...", fg="white")
        root.update()

        time_steps = 12
        X, y = create_sequences(time_series_df['Значение'].values, time_steps)
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
            LeakyReLU(alpha=0.1),
            LSTM(50, return_sequences=False),
            LeakyReLU(alpha=0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Создание ProgressBar и метки прогресса
        progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=10)
        progress_label = tk.Label(root, text="Прогресс: 0%")
        progress_label.pack()

        # Обучение модели с обновлением ProgressBar
        epochs = 100
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            history_batch = model.fit(
                X_train, y_train, 
                epochs=1, batch_size=16,
                validation_data=(X_test, y_test), 
                verbose=0
            )
            history['loss'].append(history_batch.history['loss'][0])
            history['val_loss'].append(history_batch.history['val_loss'][0])

            # Обновление ProgressBar
            progress = (epoch + 1) / epochs * 100
            progress_bar['value'] = progress
            progress_label.config(text=f"Прогресс: {int(progress)}%")
            root.update()

        # Удаление ProgressBar после завершения
        progress_bar.destroy()
        progress_label.destroy()

        # Оценка точности модели
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        messagebox.showinfo("Точность модели", f"Потери на тестовых данных: {test_loss:.4f}")

        # Построение графика потерь
        plot_loss(history['loss'], history['val_loss'])

        status_label.config(text="Обучение завершено.", fg="green")
    except Exception as e:
        status_label.config(text="Ошибка при обучении.", fg="red")
        messagebox.showerror("Ошибка", f"Не удалось обучить модель: {e}")


# Функция для загрузки ранее обученной модели
def load_trained_model():
    global model
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])
    if not file_path:
        return None
    try:
        # Из-за проблем с сериализацией нужен такой костыль
        model = legacy_h5_format.load_model_from_hdf5(file_path, custom_objects={'mse': 'mse'})
        messagebox.showinfo("Успех", "Модель успешно загружена!")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")


# Функция для сохранения модели
def save_model():
    global model
    if model is None:
        messagebox.showerror("Ошибка", "Сначала обучите или загрузите модель.")
        return None
    file_path = filedialog.asksaveasfilename(
        defaultextension=".h5", filetypes=[("Model files", "*.h5")])
    if not file_path:
        return None
    try:
        model.save(file_path)
        messagebox.showinfo("Успех", "Модель успешно сохранена!")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {e}")


# Функция для предсказания
def predict_future():
    global model, time_series_df, scaler
    if model is None or time_series_df is None:
        messagebox.showerror("Ошибка", "Сначала загрузите данные и обучите/загрузите модель.")
        return

    # Создание диалогового окна для выбора года
    year = tk.simpledialog.askinteger(
        "Выбор года", 
        "Введите год для прогноза (2025–2030):", 
        minvalue=2025, maxvalue=2030
    )
    if year is None:  # Если пользователь закрыл диалог
        return

    try:
        last_sequence = time_series_df['Значение'].values[-12:]  # Последние 12 месяцев данных
        steps = (year - 2024) * 12  # Количество шагов для прогноза (месяцы)
        
        # Предсказание значений
        predictions_normalized = predict_future_values(model, last_sequence, steps=steps)
        predictions = scaler.inverse_transform(
            np.array(predictions_normalized).reshape(-1, 1)).flatten()

        # Выборка только для выбранного года
        start_index = (year - 2025) * 12
        end_index = start_index + 12
        forecast_year = pd.DataFrame({
            'Месяц': months,
            'Прогноз': predictions[start_index:end_index]
        })

        # Сохранение результата в txt файл
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            forecast_year.to_csv(file_path, sep='\t', index=False)
            messagebox.showinfo("Успех", f"Результаты для {year} года успешно сохранены!")

        # Построение графика прогноза
        plot_forecast(forecast_year, year)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось выполнить предсказание: {e}")


# Функция для предсказания значений
def predict_future_values(model, last_sequence, steps=12):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(steps):
        prediction = model.predict(
            current_sequence[-12:].reshape(1, 12, 1), verbose=0)
        future_predictions.append(prediction[0][0])
        current_sequence = np.append(current_sequence, prediction[0][0])
    return future_predictions


# Функция для построения графика потерь
def plot_loss(train_loss, val_loss):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label='Обучающая выборка', color='blue')
    ax.plot(val_loss, label='Тестовая выборка', color='orange')
    ax.set_title('График потерь (Loss)', fontsize=14)
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_ylabel('Потери', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)


# Функция для построения графика прогноза
def plot_forecast(forecast_data, year):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Месяц', y='Прогноз', data=forecast_data,
                palette='rocket', ax=ax)
    ax.set_title(f'Прогноз на {year} год', fontsize=14)
    ax.set_xlabel('Месяц', fontsize=12)
    ax.set_ylabel('Прогнозируемое значение', fontsize=12)
    plt.xticks(rotation=45)

    # Серые надписи для предсказаний
    for index, row in forecast_data.iterrows():
        plt.text(index, row['Прогноз'] + 0.5, round(row['Прогноз'], 2),
                 ha='center', fontsize=10, color='grey')

    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close(fig)


# Функция для очистки графиков
def clear_graphs():
    for widget in left_frame.winfo_children():
        widget.destroy()
    for widget in right_frame.winfo_children():
        widget.destroy()
    messagebox.showinfo("Успех", "Графики успешно очищены!")


# Создание GUI
root = tk.Tk()
root.title("Прогноз погоды")
root.geometry("1000x600")

# Статусная строка
status_label = tk.Label(root, text="Готово", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# Кнопки
btn_load_data = ttk.Button(root, text="Загрузить данные", command=load_data)
btn_load_data.pack(pady=10)

btn_train_model = ttk.Button(root, text="Обучить модель", command=train_model)
btn_train_model.pack(pady=10)

btn_load_model = ttk.Button(root, text="Загрузить модель", command=load_trained_model)
btn_load_model.pack(pady=10)

btn_save_model = ttk.Button(root, text="Сохранить модель", command=save_model)
btn_save_model.pack(pady=10)

btn_predict = ttk.Button(root, text="Выполнить предсказание", command=predict_future)
btn_predict.pack(pady=10)

btn_clear_graphs = ttk.Button(root, text="Очистить графики", command=clear_graphs)
btn_clear_graphs.pack(pady=10)

# Область для графиков
graph_panes = tk.PanedWindow(root, orient=tk.HORIZONTAL)
graph_panes.pack(fill=tk.BOTH, expand=True)

# Левая панель для графика потерь
left_frame = tk.Frame(graph_panes, bd=2, relief=tk.SUNKEN)
graph_panes.add(left_frame, stretch="always")

# Правая панель для графика прогноза
right_frame = tk.Frame(graph_panes, bd=2, relief=tk.SUNKEN)
graph_panes.add(right_frame, stretch="always")

root.mainloop()
