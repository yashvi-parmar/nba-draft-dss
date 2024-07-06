from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

app = Flask(__name__)

# Load datasets once
lists = ['NBA18.txt', 'NBA19.txt', 'NBA20.txt', 'NBA21.txt', 'NBA22.txt', 'NBA23.txt']
df = pd.DataFrame()

for file_name in lists:
    file_path = os.path.join(file_name)
    nba_df = pd.read_csv(file_path, header=1)
    draft_year = int(file_path.split('.')[0][-2:])
    nba_df['draft_year'] = draft_year
    df = pd.concat([df, nba_df], ignore_index=True)

nba_df = df

file_path = 'nbacollege.csv'
college_df = pd.read_csv(file_path, header=0)

df = pd.merge(college_df, nba_df, left_on='player', right_on='Player')
df = df.dropna(subset=['College'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    draft_year = int(request.form['draft_year'])
    team_need = request.form['team_need']

    testing_df = df[df['draft_year'] == draft_year]
    training_df = df[df['draft_year'] != draft_year]

    # Ensure clustering and regression are re-run with new inputs
    def kmeans_clustering(df, features):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])

        kmean = KMeans(n_clusters=3, random_state=42)
        kmean.fit(scaled_features)
        cluster_labels = kmean.labels_

        df['cluster'] = cluster_labels

    features = ['mp_per_g', 'trb_per_g', 'pts_per_g', 'ast_per_g']

    kmeans_clustering(training_df, features)
    kmeans_clustering(testing_df, features)

    training_df.dropna(inplace=True)
    testing_df.dropna(inplace=True)

    X_train = training_df[features]
    y_train = training_df['WS']
    X_test = testing_df[features]
    y_test = testing_df['WS']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    testing_df['predicted_win_shares'] = y_pred

    def recommend_players(team_need):
        if team_need == 'shooter':
            cluster_label = 1
        elif team_need == 'playmaker':
            cluster_label = 2
        elif team_need == 'defender':
            cluster_label = 0
        else:
            return "Invalid team need specified."

        test = testing_df[testing_df['cluster'] == cluster_label]

        top_players = test.sort_values(by='predicted_win_shares', ascending=False).head(3)
        return top_players

    top_3_players = recommend_players(team_need)

    return render_template('results.html', players=top_3_players.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
