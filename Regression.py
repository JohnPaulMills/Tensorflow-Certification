# Import required libraries
import tensorflow as tf
import pandas as pd
import numpy as np
# Create dataset (x and y into the model)
insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
# Apply normalization on data and encoding
# Could use sklearn.preprocessing MinMaxScaler and OneHotEncoder instead.
insurance['age']=insurance['age']/insurance['age'].max()
insurance['bmi']=insurance['bmi']/insurance['bmi'].max()
insurance['children']=insurance['children']/insurance['children'].max()
insurance_encoded = pd.get_dummies(insurance)
x_numpy = insurance_encoded.drop(columns=['charges']).to_numpy()
y_numpy = insurance_encoded['charges'].to_numpy()
# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100),
    tf.keras.layers.Dense(units=10),
    tf.keras.layers.Dense(units=1)
])
model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)
history = model.fit(
    x=x_numpy,
    y=y_numpy,
    epochs=100,
)
# Do not use training data for prediction. Done in this instance to test the function.
model.predict(np.expand_dims(x_numpy[0],0)) # x_numpy[0] will fail, as the shape is (11,) and not (11,1). Shape of x_numpy is (1338, 11)
# Plot loss and accuracy
pd.DataFrame(history.history).plot()
# pd.DataFrame(history.history['loss']).plot() # individual plot
# Save and load model
model.save('saved_model/regression_model')
model_loaded = tf.keras.models.load_model('saved_model/regression_model')
print(model.summary())
print(model_loaded.summary())
