import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X = pickle.load(open('X.pkl', 'rb'))
Y = pickle.load(open('Y.pkl', 'rb'))

X = X / 255.0
X = X.reshape(-1, 224, 224, 1)

Y = to_categorical(Y, num_classes=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape =(224, 224, 1) ))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu' ))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3,3), activation='relu' ))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=32,validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_accuracy}")

model.save('model.h1')
