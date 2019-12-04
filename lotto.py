from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from openpyxl import load_workbook
import numpy as np
load_xl = load_workbook("\Study\Keras\lotto.xlsx", data_only=True)

load_sheet = load_xl['lotto']

lotto_round = np.array([int(load_sheet.cell(x,2).value) for x in range(890, 5, -1)])
lotto_num = np.array([int(load_sheet.cell(i,4).value) for i in range(890, 5, -1)])

lotto_data1 = np.array([int(load_sheet.cell(x,2).value) for x in range(890, 5, -1)])
lotto_data2 = np.array([int(load_sheet.cell(i,4).value) for i in range(890, 5, -1)])

model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))


for i in range(161, 0, -40):
    model.add(Dense(i))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # accuracy
model.fit(lotto_round, lotto_num, epochs=100, batch_size=10)

loss, acc = model.evaluate(lotto_data1, lotto_data2, batch_size=10)

print("acc :", acc)
print("loss : ", loss)

y_predict = model.predict([887])
print(y_predict)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(lotto_data2, y_predict))


r2_y_predict = r2_score(lotto_data2, y_predict)
print("R2 : ", r2_y_predict)