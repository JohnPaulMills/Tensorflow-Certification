# Import required libraries
import tensorflow as tf
# Import data
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(
        input_shape=(28, 28) # input layer (we had to reshape 28x28 to 784, the Flatten layer does this for us)
    ),
    tf.keras.layers.Dense(
        units=100,
        activation='relu'
    ),
    tf.keras.layers.Dense(
        units=10,
        activation='softmax' # for Binary-classification, use sigmoid
    )
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(
    x=train_data,
    y=train_labels,
    epochs=10,
    validation_data=(test_data, test_labels)
)
prediction = model.predict(test_data)
prediction_label = tf.argmax(prediction, axis=1)
model.save('saved_model/classification_model')
