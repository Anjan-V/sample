from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Expanded training dataset
data = pd.DataFrame({
    'team1': ['Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
              'Sunrisers Hyderabad', 'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans',
              'Mumbai Indians', 'Chennai Super Kings', 'Delhi Capitals', 'Royal Challengers Bangalore', 'Kolkata Knight Riders'],
    
    'team2': ['Delhi Capitals', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Chennai Super Kings',
              'Punjab Kings', 'Lucknow Super Giants', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Rajasthan Royals',
              'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants', 'Sunrisers Hyderabad', 'Rajasthan Royals'],
    
    'venue': ['Wankhede', 'Chepauk', 'Feroz Shah Kotla', 'M. Chinnaswamy', 'Eden Gardens',
              'Narendra Modi Stadium', 'Arun Jaitley Stadium', 'PCA Stadium', 'Ekana Stadium', 'Sawai Mansingh Stadium',
              'Wankhede', 'Chepauk', 'Feroz Shah Kotla', 'M. Chinnaswamy', 'Eden Gardens'],
    
    'weather': ['Clear', 'Clear', 'Rainy', 'Clear', 'Rainy', 'Cloudy', 'Cloudy', 'Clear', 'Rainy', 'Clear',
                'Rainy', 'Clear', 'Cloudy', 'Clear', 'Rainy'],
    
    'wickets_left': [3, 4, 2, 5, 6, 7, 8, 5, 4, 6, 3, 6, 5, 7, 4],  
    'overs_left': [5, 6, 8, 7, 4, 10, 12, 3, 6, 7, 4, 5, 6, 8, 7],    
    'current_runs': [180, 165, 145, 170, 140, 120, 130, 110, 190, 150, 160, 175, 140, 150, 180],  
    'target': [200, 190, 160, 180, 150, 140, 150, 120, 210, 175, 170, 185, 160, 175, 190],  
    'winner': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 = Batting Team Wins, 0 = Bowling Team Wins
})

# Encoding categorical values
label_encoders = {}
for column in ['team1', 'team2', 'venue', 'weather']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Store encoders for later use

# Splitting data
X = data[['team1', 'team2', 'venue', 'weather', 'wickets_left', 'overs_left', 'current_runs', 'target']]
y = data['winner']

# Standardizing numerical features
scaler = StandardScaler()
X[['wickets_left', 'overs_left', 'current_runs', 'target']] = scaler.fit_transform(X[['wickets_left', 'overs_left', 'current_runs', 'target']])

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized RandomForest Model
model = RandomForestClassifier(
    n_estimators=300,  # More trees for better learning
    max_depth=12,  # Control depth to prevent overfitting
    min_samples_split=5,  # Avoid overfitting on small data
    min_samples_leaf=2,  # Avoid overfitting on leaves
    random_state=42
)
model.fit(X_train, y_train)

# Function to safely transform new inputs
def safe_transform(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return encoder.transform([encoder.classes_[0]])[0]  # Use first class as fallback

# Function to scale numerical input before prediction
def scale_features(features):
    return scaler.transform(pd.DataFrame([features], columns=['wickets_left', 'overs_left', 'current_runs', 'target']))[0]


# HTML Code
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPL Prediction</title>
     
  <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #ECBF40;
            color: #fff;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        .container {
            max-width: 900px;
            margin: 50px auto;
            padding: 30px;
            background-color: #d6002b;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
            border: 5px solid #7C95CC;
        }

        h1 {
            text-align: center;
            font-size: 36px;
            font-family: 'Bebas Neue', sans-serif;
            color: #7C95CC;
            text-transform: uppercase;
            margin-bottom: 20px;
            animation: fadeIn 2s ease-out;
        }

        .team-logo {
            display: block;
            max-width: 150px;
            margin: 20px auto;
            transition: transform 0.3s ease;
        }

        .team-logo:hover {
            transform: scale(1.1);
        }

        button {
            padding: 15px;
            background-color: #2D56B3;
            color: #000;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 20px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #7C95CC;
            color: #fff;
        }

        form {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        label, select, input {
            font-size: 18px;
            margin-bottom: 10px;
        }

        input, select {
            padding: 12px;
            background-color: #333;
            border-radius: 8px;
            color: #fff;
            border: 2px solid #7C95CC;
        
        }
        
        .prediction {
            margin-top: 20px;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            color: #7C95CC;
        }
    
    
    </style>
</head>
<body>
    <div class="container">
        <h1>IPL Score Prediction</h1>
        <form action="/predict" method="POST">
            <label>Batting Team:</label>
            <select name="batting_team">
                <option value="Mumbai Indians">Mumbai Indians</option>
                <option value="Chennai Super Kings">Chennai Super Kings</option>
                <option value="Delhi Capitals">Delhi Capitals</option>
                <option value="Royal Challengers Bangalore">Royal Challengers Bangalore</option>
                <option value="Kolkata Knight Riders">Kolkata Knight Riders</option>
            </select>

            <label>Bowling Team:</label>
            <select name="bowling_team">
                <option value="Mumbai Indians">Mumbai Indians</option>
                <option value="Chennai Super Kings">Chennai Super Kings</option>
                <option value="Delhi Capitals">Delhi Capitals</option>
                <option value="Royal Challengers Bangalore">Royal Challengers Bangalore</option>
                <option value="Kolkata Knight Riders">Kolkata Knight Riders</option>
            </select>

            <label>Venue:</label>
            <select name="venue">
                <option value="Wankhede">Wankhede</option>
                <option value="Chepauk">Chepauk</option>
                <option value="Feroz Shah Kotla">Feroz Shah Kotla</option>
                <option value="M. Chinnaswamy">M. Chinnaswamy</option>
                <option value="Eden Gardens">Eden Gardens</option>
            </select>

            <label>Weather:</label>
            <select name="weather">
                <option value="Clear">Clear</option>
                <option value="Rainy">Rainy</option>

            </select>

            <label>Wickets Left:</label>
            <input type="number" name="wickets_left" required>

            <label>Overs Left:</label>
            <input type="number" name="overs_left" required>

            <label>Current Runs:</label>
            <input type="number" name="current_runs" required>

            <label>Target:</label>
            <input type="number" name="target" required>

            <button type="submit">Predict</button>
        </form>
        
        {% if prediction %}
        <div class="prediction">
            Prediction: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_code)

@app.route('/predict', methods=['POST'])
def predict():
    team1 = safe_transform(request.form.get('batting_team'), label_encoders['team1'])
    team2 = safe_transform(request.form.get('bowling_team'), label_encoders['team2'])
    venue = safe_transform(request.form.get('venue'), label_encoders['venue'])
    weather = safe_transform(request.form.get('weather'), label_encoders['weather'])
    wickets_left = int(request.form.get('wickets_left'))
    overs_left = int(request.form.get('overs_left'))
    current_runs = int(request.form.get('current_runs'))
    target = int(request.form.get('target'))
    
    new_match = pd.DataFrame([[team1, team2, venue, weather, wickets_left, overs_left, current_runs, target]], columns=X.columns)
    
    # Debugging print statements
    print(f"Input Data: {new_match}")

    predicted_winner = model.predict(new_match)[0]
    print(f"Predicted Winner (Raw Output): {predicted_winner}")  # Debugging

    # ✅ FIXED: Ensure prediction is correctly assigned
    if predicted_winner == 1:
        prediction = 'Batting Team Wins'  
    else:
        prediction = 'Bowling Team Wins'  # ✅ Corrected this line

    return render_template_string(html_code, prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)
