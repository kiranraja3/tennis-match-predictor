import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("C:\\Users\\kiran\\Downloads\\WryEvenModulus\\WryEvenModulus\\charting-m-matches.csv", 'r', encoding='ISO-8859-1', errors='ignore') as f:
  matches = pd.read_csv(f, engine='python', on_bad_lines='skip')

with open("C:\\Users\\kiran\\Downloads\\WryEvenModulus\\WryEvenModulus\\charting-m-points.csv", 'r', encoding='ISO-8859-1', errors='ignore') as f:
  points = pd.read_csv(f, engine='python', on_bad_lines='skip')


df_combined = pd.merge(matches, points, on="match_id", how="outer")

df_combined = df_combined.dropna(subset=["Player 1", "Player 2"])
df_combined = df_combined.drop("Charted by", axis=1)
df_combined = df_combined.drop("Notes", axis=1)

df_combined.to_csv("fulldata.csv", index=False)

def load_data(csv_file="combined.csv"):
  df = pd.read_csv(csv_file)
  return df

def preprocess_data(df, scaler=None, training=True):
    
    # Define columns to drop – identifiers, player names, and target columns
    drop_columns = ["match_id", "Player 1", "Player 2", "Charted by", "Notes", "GmW", "PtWinner","SetW", "Pt","Set1","Set2","Gm1","Gm2","Gm#","TbSet","TB?","TBpt","PtsAfter","isSvrWinner","Gm1.1","Gm2.1","Set1.1","Set2.1",
    "isRallyWinner", "isForced", "isUnforced", "isDouble"]

    # Extract targets:
    # GmW: expected to be 0, 1, or 2 (leave as-is)
    # PtWinner: remap 1 -> 0 and 2 -> 1 so that it is 0-indexed.
    if "GmW" in df.columns and "PtWinner" and "SetW" in df.columns:
        y_gmw = pd.to_numeric(df["GmW"], errors='coerce').fillna(0).astype(int)
        y_pt = pd.to_numeric(df["PtWinner"], errors='coerce').fillna(1).astype(int) - 1
        y_setw = pd.to_numeric(df["SetW"], errors='coerce').fillna(0).astype(int)
    else:
        raise ValueError("Both target columns ('GmW' and 'PtWinner') must be present in the data.")

    # Prepare features DataFrame by dropping non-feature columns.
    X = df.drop(columns=drop_columns, errors='ignore')

    # Convert all remaining columns to numeric. Non-convertible values become NaN, then fill with 0.
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale features: if training, fit the scaler; otherwise, use the provided scaler.
    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for non-training data.")
        X_scaled = scaler.transform(X)

    return X_scaled, (y_gmw, y_pt, y_setw), scaler

def get_train_test_data(csv_file="combined.csv", test_size=0.3, random_state=42):
    
    df = load_data(csv_file)
    X, y, scaler = preprocess_data(df, training=True)
    y_gmw, y_pt, y_setw = y

    # Split features and targets (using the same random state for consistency)
    X_train, X_test, y_gmw_train, y_gmw_test = train_test_split(
        X, y_gmw, test_size=test_size, random_state=random_state
    )
    _, _, y_pt_train, y_pt_test = train_test_split(
        X, y_pt, test_size=test_size, random_state=random_state
    )
    _, _, y_setw_train, y_setw_test = train_test_split(
        X, y_setw, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, (y_gmw_train, y_pt_train, y_setw_train), (y_gmw_test, y_pt_test, y_setw_test), scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = get_train_test_data()
    print("Preprocessing complete. Training data shape:", X_train.shape)