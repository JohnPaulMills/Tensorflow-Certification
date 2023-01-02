# Import required libraries
import tensorflow as tf
import pandas as pd
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
model.fit(
    x=x_numpy,
    y=y_numpy,
    epochs=100,
)
model.save('saved_model/regression_model')
print(tf.__version__)

