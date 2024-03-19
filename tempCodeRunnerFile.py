model= tf.keras.models.load_model('handwritten4.keras')
loss, accuracy=model.evalluate(x_test,y_test)