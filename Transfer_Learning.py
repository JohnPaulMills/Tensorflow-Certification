# Import required libraries
import tensorflow as tf
import tensorflow_hub as hub
# Create datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_AllData\10_food_classes_all_data\train',
    batch_size=32,
    image_size=(224,224),
    shuffle=True
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=r'C:\Users\john.mills\PycharmProjects\pythonProject\FoodVision_AllData\10_food_classes_all_data\test',
    batch_size=32,
    image_size=(224,224)
)
# Create model
model = tf.keras.Sequential([
    # For EfficientNet, input preprocessing is included as part of the model (as a Rescaling layer), and thus tf. keras
    tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True
        # if include_top is False, specify input_shape. by default, it will handle any shape.
        # Output dimension of this layer will be more than 3 if include_top=False. If so, need to set a Flatten layer. Without flatten layer the output shape will not match the label shape.
        # By setting include_top=False, the model shape is exactly the same as using hub.KerasLayer (see model1)
        #input_shape=(224,224,3)
    ),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        units=10,
        activation='softmax'
    )
])
# Create our own model
model1 = tf.keras.Sequential([
    # If not in tf.keras.applications, can call hub.KerasLayer(handle=url from tfhub.dev)
    # use the feature extraction layer as the base
    hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b0/classification/1',
                                             trainable=False,
                                             name='feature_extraction_layer',
                                             input_shape=(224,224,3)),
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
    ])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3 # Number of epochs with no improvement after which training will be stopped, default is 0. Increase to train it for longer.
)
model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=5, # set it to reduce number of batches per epoch, total runtime
    validation_data=test_ds,
    callbacks=[callback]
    #validation_steps=5 # set it to reduce number of val batches per epoch, toral runtime.
)
model.save('saved_model/CNN_TransferLearning_model')