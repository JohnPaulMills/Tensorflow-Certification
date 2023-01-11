import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import wget
import pandas as pd

# # Download text to local repository
# filename = wget.download('https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip')
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall('nlp_getting_started')
train_df = pd.read_csv(r'nlp_getting_started\train.csv')
train_text_numpy = train_df['text'][:7000].to_numpy()
train_target_numpy = train_df['target'][:7000].to_numpy()
test_text_numpy = train_df['text'][7000:].to_numpy()
test_target_numpy = train_df['target'][7000:].to_numpy()
# Turn data into TensorFlow Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_text_numpy,train_target_numpy))
test_ds = tf.data.Dataset.from_tensor_slices((test_text_numpy,test_target_numpy))
# Take the TensorSliceDataset's and turn them into prefetched batches
train_batch = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
test_batch = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
# Build model
inputs = tf.keras.layers.Input(
    shape=(1,),  # inputs are 1-dimensional strings
    dtype="string")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=15
)
vectorizer.adapt(train_text_numpy)
vectorizer = vectorizer(inputs)


# And to make sure we're not getting reusing trained embeddings,we'll create another embedding layer for each model.
# The text_vectorizer layer can be reused since it doesn't get updated during training.


def create_embedding_layer(vectorizerlayer):
    embeddinglayer = tf.keras.layers.Embedding(
        input_dim=10000,
        output_dim=128,
        input_length=15
    )(vectorizerlayer)
    return embeddinglayer


embedding = create_embedding_layer(vectorizer)
pooling = tf.keras.layers.GlobalAveragePooling1D()(embedding)
dense = tf.keras.layers.Dense(
    units=1, # For multi-class, unit = number of classes
    activation='sigmoid' # For multi-class, softmax
)
outputs = dense(pooling)
model_dense = tf.keras.Model(inputs, outputs)


def compile_and_fit(model):
    model.compile(
        # For multi-class, if one hot form use "categorical_crossentropy", if int form use sparse_categorical_crossentropy
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    model.fit(
        x=train_text_numpy,
        y=train_target_numpy,
        epochs=5,
        # steps_per_epoch=5,
        validation_data=(test_text_numpy, test_target_numpy)
    )
    return model


compile_and_fit(model_dense)

# Build a model with lstm layer
embedding = create_embedding_layer(vectorizer)
# x = tf.keras.layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
lstm = tf.keras.layers.LSTM(64)(embedding) # return vector for whole sequence
# x = tf.keras.layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(lstm)
model_lstm = tf.keras.Model(inputs, outputs)
compile_and_fit(model_lstm)

# Build a model with GRU layer
embedding = create_embedding_layer(vectorizer)
# x = layers.GRU(64, return_sequences=True)(x)
gru = tf.keras.layers.GRU(64)(embedding) # return vector for whole sequence
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(gru)
model_gru = tf.keras.Model(inputs, outputs)
compile_and_fit(model_gru)

# Build a model with Bi-Directional RNN layer
embedding = create_embedding_layer(vectorizer)
# x = tf.keras.layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding) # return vector for whole sequence
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(bidirectional)
model_bidirectional = tf.keras.Model(inputs, outputs)
compile_and_fit(model_bidirectional)

# Build a model with Conv1D layer
embedding = create_embedding_layer(vectorizer)
conv_1d = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu")(embedding) # convolve over target sequence 5 words at a time
max_pool = tf.keras.layers.GlobalMaxPool1D()(conv_1d)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(max_pool)
model_conv1d = tf.keras.Model(inputs, outputs)
compile_and_fit(model_conv1d)

# Build a model with TensorFlow Hub Pretrained Sentence Encoder
model_hub = tf.keras.Sequential([
    # hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4"), # Doesn't load, hence used a different encoder below.
    hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim50/2'),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_6_USE")
compile_and_fit(model_hub)

# Predict
prediction=tf.round(model_hub.predict(test_text_numpy))
