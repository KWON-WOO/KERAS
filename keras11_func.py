from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5
)


# model = Sequential()
# model.add(Dense(200, input_shape=(1,), activation='relu'))
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# dense = []

# input1 = Input(shape=(1,))
# dense.append(Dense(5, activation='relu')(input1))

# for i in range(1,20):
#     dense.append(Dense(i)(dense[i-1]))
# dense.append(Dense(1)(dense[19]))

# model = Model(inputs = input1, outputs = dense[20])
# for i in range(151, 0, -50):
#     model.add(Dense(i))

input1 = Input(shape=(1,))
xx = Dense(5, activation='relu')(input1)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
output1 = Dense(1)(xx)

model = Model(inputs = input1, outputs = output1)
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

