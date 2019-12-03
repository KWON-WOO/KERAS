from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([x for x in range(1,11)])
y = np.array([x for x in range(1,11)])
x2 = np.array([x for x in range(11,16)])

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))

for i in range(91, 0, -10):
    model.add(Dense(i))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y,epochs=100)

loss, acc = model.evaluate(x,y)
print("acc :", acc)
print("loss : ", loss)

y_predict = model.predict(x2)
print(y_predict)