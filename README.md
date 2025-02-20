Tennis Match Predictor

This project is a RNN based model designed to predict the outcome of tennis matches down to the set and point level. The model leverages machine learning techniques to analyze historical match data and make informed predictions.

Features

Predicts set, point, and game outcomes based on historical data

Analyzes set-by-set and point-by-point results

Utilizes deep learning for enhanced accuracy



Download Dataset

The dataset is too large to be stored in this repository. Please download it from the following link and place it in the data/ directory:

[Download Dataset](https://www.kaggle.com/datasets/ryanthomasallen/tennis-match-charting-project)

Installation

Clone the repository:

git clone https://github.com/kiranraja3/tennis-match-predictor.git
cd tennis-match-predictor

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Training the Model

Ensure the dataset is in the data/ directory.

Train the model/preprocess data:

python src/train.py

Making Predictions

After training, use the following command to make predictions:

python src/predict.py --input test_match_data.csv

License

This project is licensed under the MIT License.
