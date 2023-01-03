# Import required libraries
import tensorflow as tf
import zipfile
import wget
# Download images to local repository
'''filename = wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('FoodVision_PizzaSteak')'''
# Create dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_PizzaSteak\pizza_steak\train',
  image_size=(224, 224),
  batch_size=32,
  shuffle=True)
test_ds = tf.keras.utils.image_dataset_from_directory(
  directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_PizzaSteak\pizza_steak\test',
  image_size=(224, 224),
  batch_size=32)
# Create model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(),
  optimizer=tf.keras.optimizers.Adam(),
  metrics=['accuracy']
)
model.fit(
  train_ds,
  epochs=5,
  validation_data=test_ds
)
model.save('saved_model/CNN_Binary_model')

