from keras.models import Sequential
from keras.layers import Dense
import numpy as np
lotto_date = np.array(887,886,885,884,883,882,881,880,879,878,877)
lotto_answer = np.array([8,14,17,27,36,45,10],
[19,23,28,37,42,45,2],[1,3,24,27,39,45,31],[4,14,23,28,37,45,17],
[9,18,32,33,37,44,22],[18,34,39,43,44,45,23],[4,18,20,26,27,32,9],)
x_train = np.array([x for x in range(1,11)])
y_train = np.array([x for x in range(1,11)])
x_test = np.array([x for x in range(11,21)])
y_test = np.array([x for x in range(11,21)])
x_predict = np.array([x for x in range(21,26)])

model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))

for i in range(181, 0, -20):
    model.add(Dense(i))

# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=1)

loss, acc = model.evaluate(x_test,y_test, batch_size=1)

print("acc :", acc)
print("loss : ", loss)

y_predict = model.predict(x_predict)
print(y_predict)