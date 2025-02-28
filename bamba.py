from flask import Flask, request, redirect, url_for, flash, session
import os
from flask import Flask, request, redirect, url_for, flash, session, render_template

from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Utilisez un secret pour la gestion des sessions
app.config['UPLOAD_FOLDER'] = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/'  # Remplacez par votre chemin d'upload

MODEL_PATH = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl'  # Remplacez par le chemin de votre modèle
@app.route('/form')
def form():
    return render_template('form.html')
@app.route('/load_model_and_predict', methods=['POST'])
def load_model_and_predict():
    # Vérifier si le fichier a été envoyé
    if 'file' not in request.files:
        flash("Aucun fichier téléchargé", "danger")
        return redirect(url_for('form'))  # Retourner à la page du formulaire

    # Récupérer le fichier téléchargé
    file = request.files['file']

    if file:
        # Sécuriser le nom du fichier
        filename = secure_filename(file.filename)

        # Définir le chemin où le fichier sera sauvegardé
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Sauvegarder le fichier sur le disque
        file.save(file_path)

        # Charger le modèle pré-entraîné
        model = joblib.load(MODEL_PATH)

        # Charger le fichier CSV dans un DataFrame
        new_data = pd.read_csv(file_path)

        # Prétraitement des données (par exemple, calculer la durée du vol)
        if 'departure_time' in new_data.columns and 'arrival_time' in new_data.columns:
            new_data['departure_time'] = pd.to_datetime(new_data['departure_time'])
            new_data['arrival_time'] = pd.to_datetime(new_data['arrival_time'])
            new_data['flight_duration'] = (new_data['arrival_time'] - new_data['departure_time']).dt.total_seconds() / 60
        else:
            new_data['flight_duration'] = np.nan

        # Ajouter la colonne 'live' si elle n'existe pas et s'assurer qu'elle est de type numérique
        if 'live' not in new_data.columns:
            new_data['live'] = 0

        # Remplacer les valeurs NaN dans 'live' par 0
        new_data['live'] = new_data['live'].fillna(0)

        # Convertir 'live' en type entier
        new_data['live'] = new_data['live'].astype(int)

        # Encodage des autres colonnes avec LabelEncoder
        le_flight = LabelEncoder()
        le_airline = LabelEncoder()
        le_flight_status = LabelEncoder()
        le_departure_airport = LabelEncoder()
        le_arrival_airport = LabelEncoder()

        # Transformation des colonnes catégorielles
        new_data['flight'] = le_flight.fit_transform(new_data['flight'].astype(str))
        new_data['airline'] = le_airline.fit_transform(new_data['airline'].astype(str))
        new_data['flight_status'] = le_flight_status.fit_transform(new_data['flight_status'].astype(str))
        new_data['departure_airport'] = le_departure_airport.fit_transform(new_data['departure_airport'].astype(str))
        new_data['arrival_airport'] = le_arrival_airport.fit_transform(new_data['arrival_airport'].astype(str))

        # Préparer les données pour la prédiction
        X_new = new_data[['flight_status', 'airline', 'flight', 'live', 'departure_airport', 'arrival_airport', 'flight_duration']]

        # Prédiction avec le modèle
        predictions = model.predict(X_new)

        # Ajouter les prédictions à la colonne 'predicted_needs_parking'
        new_data['predicted_needs_parking'] = np.where(
            predictions == 1,
            "Cet avion nécessite un stationnement sur le tarmac",
            "Cet avion ne nécessite pas de stationnement sur le tarmac"
        )

        # Sauvegarder les résultats dans la session
        session['predictions_data'] = new_data[['flight_date', 'flight', 'airline', 'flight_status', 'predicted_needs_parking']].to_dict(orient='records')

        # Message de confirmation
        flash("Prédictions générées avec succès.", "success")
        return redirect(url_for('prediction_view'))

    else:
        flash("Erreur lors de l'envoi du fichier", "danger")
        return redirect(url_for('form'))


@app.route('/prediction_view')
def prediction_view():
    # Vérifier si les données de prédiction sont dans la session
    if 'predictions_data' in session:
        predictions_data = session['predictions_data']
        return render_template('prediction_view.html', predictions_data=predictions_data)
    else:
        flash("Aucune prédiction disponible.", "warning")
        return redirect(url_for('form'))


if __name__ == "__main__":
    app.run(debug=True)
