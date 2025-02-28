from flask import Flask, render_template
import pandas as pd
import numpy as np
import pickle
import io
import sys

app = Flask(__name__)

# Charger le modèle d'optimisation
model_path = '/Users/benothmane/Desktop/flight_dashboard/models/new_model_new3.pkl'  # Mettez le chemin de votre modèle ici
model = pickle.load(open(model_path, 'rb'))

# Charger les label_encoders pour les transformations
label_encoders = pickle.load(open('/Users/benothmane/Desktop/flight_dashboard/label_encoders.pkl', 'rb'))  # Ajustez le chemin ici

# Charger les données (assurez-vous que votre fichier est correct)
df = pd.read_csv('/Users/benothmane/Desktop/flight_dashboard/upload/folder/NEW_vols.csv')

# Préparez les données de test (adaptez cette partie selon votre modèle)
X = df.drop('target', axis=1, errors='ignore')  # Supposons que 'target' est la colonne à prédire
y = df['target'] if 'target' in df.columns else None  # Adapter selon votre dataset
y_pred = model.predict(X)

# Fonction pour capturer les prints et les afficher sur la page
def capture_prints(func, *args, **kwargs):
    # Créer un buffer pour capturer les prints
    buffer = io.StringIO()
    sys.stdout = buffer  # Rediriger stdout vers le buffer

    # Exécuter la fonction
    func(*args, **kwargs)

    # Récupérer le contenu du buffer
    output = buffer.getvalue()

    # Rétablir la sortie standard
    sys.stdout = sys.__stdout__

    return output

# Fonction de prédiction du parking
def predict_parking(df, y_pred):
    results = df.copy()
    results['needs_parking'] = y_pred

    # Décodage des variables catégorielles pour les résultats
    results['airline'] = label_encoders['airline'].inverse_transform(results['airline'].astype(int))
    results['flight_status'] = label_encoders['flight_status'].inverse_transform(results['flight_status'].astype(int))
    results['departure_airport'] = label_encoders['departure_airport'].inverse_transform(results['departure_airport'].astype(int))
    results['arrival_airport'] = label_encoders['arrival_airport'].inverse_transform(results['arrival_airport'].astype(int))

    optimized_results = []
    log_messages = []  # Liste pour stocker les messages de log

    for index, row in results.iterrows():
        result_entry = {
            "Airline": row['airline'],
            "Flight ID": row['flight'],
            "Needs Parking": None
        }

        log_messages.append(f"Analyse du vol : {row['flight']} de la compagnie {row['airline']}")

        if row['needs_parking'] == 1:
            log_messages.append(f"L'Avion de {row['airline']} (ID: {row['flight']}) ..stationnement autorisé...")
            result_entry["Needs Parking"] = "Peut stationner"
        else:
            log_messages.append(f"L'Avion de {row['airline']} (ID: {row['flight']}) Stationnement non autorisé!!!!")
            # Logique d'optimisation supplémentaire pour les heures de pointe
            if row['arrival_hour'] in [7, 8, 17, 18]:  # Heures de pointe
                log_messages.append("Attention : l'atterrissage aura eu lieu pendant les heures de pointe, le stationnement pourrait être compliqué!!!!!!")
                result_entry["Needs Parking"] = "Peut nécessiter un stationnement en raison de l'heure de pointe."
            else:
                result_entry["Needs Parking"] = "Ne peut pas stationner!!!!!!"

        optimized_results.append(result_entry)

    return optimized_results, log_messages


@app.route('/')
def home():
    # Appliquer la prédiction de parking et capturer les messages print
    optimized_results, log_messages = capture_prints(predict_parking, df, y_pred)

    # Convertir la liste des résultats optimisés en DataFrame pour faciliter l'affichage dans le template
    df_optimized_results = pd.DataFrame(optimized_results)

    # Passer les résultats et les messages au template HTML pour les afficher
    return render_template('index.html',
                           tables=[df_optimized_results.to_html(classes='data')],
                           titles=df_optimized_results.columns.values,
                           log_messages=log_messages)


if __name__ == '__main__':
    app.run(debug=True)
