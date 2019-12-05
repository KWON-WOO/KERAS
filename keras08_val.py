from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([x for x in range(1,11)])
y_train = np.array([x for x in range(1,11)])
x_test = np.array([x for x in range(11,21)])
y_test = np.array([x for x in range(11,21)])
x_val = np.array([x for x in range(101,106)])
y_val = np.array([x for x in range(101,106)])
# y_val = np.array([901,19897,1543,1234,11135])
# 0.9999994179801764
# 0.99999999209056
# 0.999999780181712
model = Sequential()
model.add(Dense(200, input_shape=(1,), activation='relu'))

for i in range(161, 0, -40):
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
