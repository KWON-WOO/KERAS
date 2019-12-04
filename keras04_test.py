from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = np.array([x for x in range(1,11)])
y_train = np.array([x for x in range(1,11)])
x_test = np.array([x for x in range(11,21)])
y_test = np.array([x for x in range(11,21)])
model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))

for i in range(161, 0, -40):
    model.add(Dense(i))

# model.add(Dense(5))
# model.add(Dense(3))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

loss, acc = model.evaluate(x_test,y_test, batch_size=1)

print("acc :", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)