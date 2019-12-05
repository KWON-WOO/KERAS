from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)])
print(x)

print(x.shape)
x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)

model = Sequential()
model.add(Dense(27100, input_shape=(2,), activation='relu'))
model.add(Dense(101))
model.add(Dense(12030))
model.add(Dense(101))
model.add(Dense(28200))
model.add(Dense(101))
model.add(Dense(12300))
model.add(Dense(101))
model.add(Dense(18020))
model.add(Dense(101))
model.add(Dense(12300))
model.add(Dense(101))
model.add(Dense(18200))
model.add(Dense(101))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # accuracy
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_data=(x_val, y_val))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

aaa = np.array([[101,102,103], [201,202,203]])
# aaa = np.transpose(aaa)
# print("acc :", acc)
# print("loss : ", loss)
# aaa = np.transpose(aaa)
y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
