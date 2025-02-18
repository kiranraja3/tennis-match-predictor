import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def create_model(input_dim):
    # Define the input layer
    inputs = Input(shape=(input_dim,), name="input_layer")

    # Common hidden layers
    x = Dense(128, activation='relu', name="dense_1")(inputs)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Dense(64, activation='relu', name="dense_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    x = Dense(32, activation='relu', name="dense_3")(x)

    # Three separate outputs:
    # Output for predicting GmW (3 classes: 0, 1, or 2)
    gmw_output = Dense(3, activation='softmax', name="gmw_output")(x)
    # Output for predicting PtWinner (2 classes: 0 or 1)
    ptwinner_output = Dense(2, activation='softmax', name="ptwinner_output")(x)
    # Output for predicting SetW (2 classes: 0 or 1)
    setw_output = Dense(2, activation='softmax', name="setw_output")(x)

    # Create the multi-output model
    model = Model(inputs=inputs, outputs=[gmw_output, ptwinner_output, setw_output], name="tennis_multi_output_model")

    # Compile the model using Adam optimizer and sparse categorical crossentropy for each output.
    model.compile(
        optimizer='adam',
        loss={
            'gmw_output': 'sparse_categorical_crossentropy',
            'ptwinner_output': 'sparse_categorical_crossentropy',
            'setw_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'gmw_output': 'accuracy',
            'ptwinner_output': 'accuracy',
            'setw_output': 'accuracy'
        }
    )
    return model

if __name__ == "__main__":
    # Test the model with a dummy input dimension (adjust as needed)
    model = create_model(input_dim=input_dim)
    model.summary()