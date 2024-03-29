import os
import cv2 #computer vision load and process images
import numpy as np # working with numpy arrays
import matplotlib.pyplot as plt # visualizatin of the actual digit
import tensorflow as tf # for machine learning


# mnist = tf.keras.datasets.mnist 
# #we are going to split the datasets info training data and testing data
# (x_train, y_train), (x_test,y_test) = mnist.load_data()

# x_train= tf.keras.utils.normalize(x_train,axis=1)
# x_test = tf.keras.utils.normalize(x_test , axis=1)

# # create the model , nuaral network
# model = tf.keras.models.Sequential()
# # layers
# model.add(tf.keras.layers.Flatten(input_shape=(28,28))) # Flatten layer multiply the pixals 28 * 28 and make the in one line instead of grid
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))
# #compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# # train the model 
# model.fit(x_train, y_train,epochs=20)
# model.save('handwritten4.keras')

# test the model
model= tf.keras.models.load_model('handwritten4.keras')
# loss, accuracy=model.evaluate(x_test,y_test)



image_number=1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction= model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1

