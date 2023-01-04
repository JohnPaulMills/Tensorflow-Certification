# Import required libraries
import tensorflow as tf
import zipfile
import wget

# Binary Classification
# Download images to local repository
# filename = wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip')
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall('FoodVision_PizzaSteak')
# Create dataset
train_ds_binary = tf.keras.utils.image_dataset_from_directory(
  directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_PizzaSteak\pizza_steak\train',
  image_size=(224, 224),
  batch_size=32,
  shuffle=True)
test_ds_binary = tf.keras.utils.image_dataset_from_directory(
  directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_PizzaSteak\pizza_steak\test',
  image_size=(224, 224),
  batch_size=32)
# Create model
model_binary = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)), # first layer needs an input shape specified, whether it is Rescale or Conv2D layer.
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'), # can stack Conv2D with another Conv2D
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model_binary.compile(
  loss=tf.keras.losses.BinaryCrossentropy(),
  optimizer=tf.keras.optimizers.Adam(),
  metrics=['accuracy']
)
model_binary.fit(
  train_ds_binary,
  epochs=5,
  validation_data=test_ds_binary
)
model_binary.save('saved_model/CNN_Binary_model')

# Multi-Class Classification
# Download images to local repository
# filename = wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip')
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall('FoodVision_AllData')
# Create dataset
train_ds_multi = tf.keras.utils.image_dataset_from_directory(
    directory = r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_AllData\10_food_classes_all_data\train',
    batch_size=32,
    image_size=(224,224),
    shuffle=True)
test_ds_multi = tf.keras.utils.image_dataset_from_directory(
    directory = r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_AllData\10_food_classes_all_data\test',
    batch_size=32,
    image_size=(224,224))
model_multi = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224,224,3)),
    tf.keras.layers.RandomZoom(height_factor=(-0.3, 0.3)), # added data augmentation to prevent overfitting (train accuracy up, val accurary down)
    tf.keras.layers.RandomRotation(factor=(-0.3, 0.3)), # added data augmentation to prevent overfitting (train accuracy up, val accurary down)
    # tf.keras.layers.Conv2D( # Commented out this infrastructure as complex layers can lead to overfitting
    #     filters=10,
    #     kernel_size=3,
    #     activation='relu'
    # ),
    # tf.keras.layers.Conv2D(
    #     filters=10,
    #     kernel_size=3,
    #     activation='relu'
    # ),
    # tf.keras.layers.MaxPool2D(),
    # tf.keras.layers.Conv2D(
    #     filters=10,
    #     kernel_size=3,
    #     activation='relu'
    # ),
    # tf.keras.layers.MaxPool2D(),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(30, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(10, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_multi.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model_multi.fit(
    train_ds_multi,
    epochs=5,
    validation_data=test_ds_multi
)
model_multi.save('saved_model/CNN_MultiClass_model')


