from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
from utils import CategoricalTransformer, NumericalTransformer
from tensorflow.keras import layers
import tensorflow as tf

# Your API definition
app = Flask(__name__)


@app.route("/predict", methods=['GET', 'POST'])  # use decorator pattern for the route
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_).reindex(fill_value=0)
            print(query)
            # Fit your data on the scaler object
            # scaler = StandardScaler()
            # scaled_df = scaler.fit_transform(query)
            # query = pd.DataFrame(scaled_df, columns=model_columns)
            # print(query)
            transformed_query = pipeline.transform(query)

            prediction = model.predict(transformed_query)
            print({'probability of success': str(prediction)})
            return jsonify({'probability of success': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')

def create_model():
  model = tf.keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(76,), kernel_initializer='glorot_uniform'),
  layers.Dropout(rate=0.5),
  layers.Dense(1, activation='sigmoid')
])

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'],
                     optimizer=tf.optimizers.Adam())

  return model


pipeline = joblib.load('./data/transformed_type2/obj_pipeline.pkl')

# Create a basic model instance
model = create_model()

model.load_weights('./models_type2_allpandas/checkpointbest.h5')


app.run(port=12345, debug=True)
