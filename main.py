import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import matplotlib.pyplot as plt

# загрузка данных в pandas dataframe
df = pd.read_csv('Session.csv')

for i in range(1, 10):
    df[f'user{i}'] = pd.to_datetime(df[f'user{i}']).dt.normalize()

print(df)

df2 = df.pivot_table(index='user1', aggfunc='size')
df2.describe()
df2.plot()
plt.show()

# преобразуем дату в формат float
for col in df.columns:
    df[col] = (df[col] - df[col].min()) / np.timedelta64(1, 'D')
    df[col] = df[col].apply(np.floor)

print(df)

# создаем новый датафрейм для обучения модели
data = pd.DataFrame()
# добавляем столбцы с разницами между днями для каждого пользователя и метками смены модели
for i in range(1, 10):
    if i == 9:  # исключаем 9-го пользователя из обучения
        continue
    diff = df[f'user{i}'].diff().fillna(0)
    data[f'user{i}_diff'] = diff
    change = (diff > 15).astype(int) # 43 работает, но граничные данные, 44 нет
    data[f'user{i}_change'] = (change.cumsum() >= 1).astype(int)

print(data)

data.to_csv('results.csv')
# целевые значения - столбец с метками смены модели для всех пользователей, за исключением 9-го
target = data[[f'user{i}_change' for i in range(1, 9)]].values

# данные - столбцы с разницами между днями для всех пользователей, за исключением 9-го
data = data[[f'user{i}_diff' for i in range(1, 9)]].values.reshape(-1, 8, 1)

# создание модели LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(None, 1)))
model.add(Dense(8, activation='sigmoid'))

# компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# обучение модели
history = model.fit(data, target, epochs=100, batch_size=16)

# строим график функции потерь
plt.plot(history.history['loss'], label='train_loss')
plt.legend()
plt.show()

# строим график точности
plt.plot(history.history['accuracy'], label='train_acc')
plt.legend()
plt.show()

# создаем новый датафрейм для предсказания

user9_diff = df['user9'].diff().fillna(0)
user_data = pd.DataFrame()
user_data['user9_diff'] = user9_diff
user_data.to_csv('results3.csv')
user_data = user_data['user9_diff'].values.reshape(1, -1, 1)
# находим смену модели
user_pred = model.predict(user_data)
print(user_pred[0])
# вывод результата
print(user_pred[0].max().round())
