import os
import cv2 #computer vision load and process images
import numpy as np # working with numpy arrays
import matplotlib.pyplot as plt # visualizatin of the actual digit
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # Reshape the data to include channel dimension (grayscale)
# train_images = train_images.reshape((60000, 28, 28, 1))
# test_images = test_images.reshape((10000, 28, 28, 1))

# # Scale the images to the [0, 1] range
# train_images, test_images = train_images / 255.0, test_images / 255.0


from tensorflow.keras import layers, models

# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# model.summary()

# #compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # train model
# history = model.fit(train_images, train_labels, epochs=50, validation_split=0.1)
# model.save('stest.keras')

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f"Test accuracy: {test_acc}")


model= models.load_model('stest.keras')
image_number=1
while os.path.isfile(f"digits2/{image_number}.png"):
    try:
        img = cv2.imread(f"digits2/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction= model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number+=1
