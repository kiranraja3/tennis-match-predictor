import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
from preprocess import load_data, preprocess_data

def predict_new_data(input_csv="combined.csv", model_path="tennis_model.h5", scaler_path="scaler.pkl"):
   
    df = load_data(input_csv)

    # Load the saved scaler
    scaler = joblib.load(scaler_path)

   
    drop_columns = ["match_id", "Player 1", "Player 2", "Charted by", "Notes", "GmW", "PtWinner", "SetW"]
    features = df.drop(columns=drop_columns, errors='ignore')
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = scaler.transform(features)

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Make predictions for all three outputs
    predictions = model.predict(X)
    # predictions is a list: [gmw_predictions, ptwinner_predictions, setw_predictions]
    df["Predicted GmW"] = np.argmax(predictions[0], axis=1)
    # For PtWinner and SetW, add 1 to map back to original {1,2}
    df["Predicted PtWinner"] = np.argmax(predictions[1], axis=1) + 1
    df["Predicted SetW"] = np.argmax(predictions[2], axis=1) + 1

    return df

if __name__ == "__main__":
    df_predictions = predict_new_data(input_csv="combined.csv")
    print(df_predictions.head())
    df_predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
