import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import get_train_test_data
from model import create_model
import joblib

def train_model(csv_file="fulldata.csv", model_save_path="tennis_model.h5", scaler_save_path="scaler.pkl"):
    # Load and split the data
    X_train, X_test, y_train, y_test, scaler = get_train_test_data(csv_file)
    y_gmw_train, y_pt_train, y_set_train = y_train
    y_gmw_test, y_pt_test, y_set_test = y_test

    # Build the model based on the number of features
    input_dim = X_train.shape[1]
    model = create_model(input_dim)

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with all three targets
    history = model.fit(
        X_train,
        {"gmw_output": y_gmw_train, "ptwinner_output": y_pt_train, "setw_output": y_set_train},
        validation_data=(X_test, {"gmw_output": y_gmw_test, "ptwinner_output": y_pt_test, "setw_output": y_set_test}),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop]
    )

    # Evaluate the model on the test set
    eval_results = model.evaluate(X_test, {"gmw_output": y_gmw_test, "ptwinner_output": y_pt_test, "setw_output": y_set_test})
    print("Evaluation results:", eval_results)

    # Save the trained model and scaler
    model.save(model_save_path)
    print("Model saved to", model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print("Scaler saved to", scaler_save_path)

if __name__ == "__main__":
    train_model()