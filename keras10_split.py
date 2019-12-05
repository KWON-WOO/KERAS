from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5
)

model = Sequential()
model.add(Dense(200, input_shape=(1,), activation='relu'))

for i in range(151, 0, -50):
    model.add(Dense(i))

# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # accuracy
model.fit(x_train,y_train,epochs=100,batch_size=1, validation_data=(x_val, y_val))

loss, acc = model.evaluate(x_test, y_test, batch_size=1)

print("acc :", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test,y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
