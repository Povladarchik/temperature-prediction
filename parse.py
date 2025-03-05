import requests
from bs4 import BeautifulSoup

# URL страницы, с которой будем извлекать данные
url = "http://www.pogodaiklimat.ru/history/27612.htm" # Moscow

try:
    # Отправляем GET-запрос к указанному URL
    response = requests.get(url)
    response.raise_for_status()  # Проверяем успешность запроса
except requests.exceptions.RequestException as e:
    print(f"Ошибка при запросе данных: {e}")
    exit()

# Создаем объект BeautifulSoup для парсинга HTML-контента
soup = BeautifulSoup(response.text, 'html.parser')

# Находим блоки с годами и температурными данными
year_table = soup.find('div', class_='chronicle-table-left-column')  # Блок с годами
temperature_table = soup.find('div', class_='chronicle-table')  # Блок с температурами

# Если не найден блок с годами, выводим сообщение об ошибке
if not year_table:
    print("Блок с годами (класс 'chronicle-table-left-column') не найден.")
    exit()

# Если не найден блок с температурами, выводим сообщение об ошибке
if not temperature_table:
    print("Блок с температурами (класс 'chronicle-table') не найден.")
    exit()

# Извлекаем список лет из блока с годами
years = [td.get_text(strip=True) for td in year_table.find_all('td')[1:]]

# Определяем список месяцев для маппинга данных
months = ['Год', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

# Создаем пустой список для хранения строк данных
data_rows = [months]  # Добавляем заголовок таблицы

# Проходим по строкам таблицы температур
rows = temperature_table.find_all('tr')[1:]  # Пропускаем заголовок таблицы
for i, row in enumerate(rows):
    if i >= len(years):  # Если количество строк превышает количество лет, завершаем цикл
        break

    # Текущий год
    year = years[i]

    # Извлекаем значения температур для каждого месяца
    temperatures = [td.get_text(strip=True) for td in row.find_all('td')[:-1]]

    # Создаем строку данных для текущего года
    data_row = [year] + temperatures
    data_rows.append(data_row)

# Преобразуем данные в формат строки для записи в файл
output = "\n".join([",".join(row) for row in data_rows])

# Сохраняем данные в TXT-файл
with open('files/climate_data.txt', 'w', encoding='utf-8') as txt_file:
    txt_file.write(output)

print("Данные сохранены в файл 'climate_data.txt'")