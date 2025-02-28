#importation des bibliotheques
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Force l'utilisation du backend 'Agg' pour éviter l'interface graphique
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kaleido
import plotly.graph_objects as go
import xgboost as xgb
from werkzeug.utils import secure_filename
from datetime import datetime
import pulp
import json
from flask import Response
import seaborn as sns
#_________________________________________________________
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Charger le modèle pré-entraîné
model_path = "/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl"
model = joblib.load(model_path)  # Utilisez joblib.load pour charger le modèle

# Variables pour stocker les datasets
main_dataset = None
predict_dataset = None

#________LOGIN_______________
# Route de connexion
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "OTHMANE_BAMBA" and password == "password":  # Exemple d'authentification
            session['logged_in'] = True
            return redirect(url_for('home'))
    return render_template('login.html')

# Route de déconnexion
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))
#_____________PAGE D ACCUEIL________________________
# Route de la page d'accueil
@app.route('/home', methods=['GET'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')



#___________AFFICHAGE  DE DONNEES sur Smart_parking_________________
# Configuration du dossier de sauvegarde des fichiers
app.config['UPLOAD_FOLDER'] = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'
# Nombre de lignes par page
ROWS_PER_PAGE = 6
# Route pour afficher et analyser le fichier CSV principal
# Route pour afficher et analyser le fichier CSV principal
@app.route('/smart_parking', methods=['GET', 'POST'])
def smart_parking():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    # Initialisation des variables
    csv_data = []
    page = request.args.get('page', 1, type=int)  # Page actuelle, par défaut 1
    rows_per_page = 6  # Nombre de lignes par page

    # Si un fichier CSV est téléchargé
    if request.method == 'POST' and 'dataset' in request.files:
        csv_file = request.files['dataset']
        if csv_file:
             # Sauvegarde du fichier CSV
            csv_filename = 'flights.csv'
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
            csv_file.save(csv_path)
            session['csv_path'] = csv_path  # Enregistrer le chemin du fichier dans la session
            flash('Fichier CSV téléchargé avec succès !')

# Charger les données pour pagination
    if 'csv_path' in session:
        csv_path = session['csv_path']
        try:
            data = pd.read_csv(csv_path)
            session['csv_length'] = len(data)
            start = (page - 1) * rows_per_page
            end = start + rows_per_page
            csv_data = data.iloc[start:end].to_dict(orient='records')
            total_pages = (len(data) // rows_per_page) + (1 if len(data) % rows_per_page > 0 else 0)
        except Exception as e:
            flash(f"Erreur lors de la lecture du fichier CSV : {str(e)}", "danger")
            total_pages = 0
    else:
        total_pages = 0

    return render_template('smart_parking.html', csv_data=csv_data, page=page, total_pages=total_pages)

# ______Route pour sauvegarder les données CSV_____
@app.route('/save_csv', methods=['POST'])
def save_csv():
    csv_data = request.form.get('csv_data')
    if csv_data:
        # Convertir les données JSON en DataFrame et sauvegarder le fichier CSV
        df = pd.read_json(csv_data)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'saved_flights.csv')
        df.to_csv(save_path, index=False)
        flash("Fichier CSV sauvegardé avec succès !", "success")
    else:
        flash("Erreur : Aucun contenu à sauvegarder.", "danger")
    return redirect(url_for('smart_parking'))

    #_______ Route de retour à la page d'accueil________
    @app.route('/home', methods=['GET', 'POST'])
    def home_view():
        csv_files = []
        files_path = app.config['UPLOAD_FOLDER']
        if os.path.exists(files_path):
            csv_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
        section = request.args.get('section', 'accueil')
        return render_template('home.html', csv_files=csv_files, section=section)

#________________DATASET TELECHARGES_________________

# Définir le chemin vers le dossier de téléchargement
app.config['UPLOAD_FOLDER'] = "/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder"

@app.route('/dataset_view')
def dataset_view():
    # Récupérer la liste des fichiers CSV
    files_path = app.config['UPLOAD_FOLDER']
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]

    # Récupérer le nom du fichier pour l'aperçu
    selected_file = request.args.get('file')
    file_preview = None
    current_page = int(request.args.get('page', 1))  # Page actuelle (par défaut 1)

    if selected_file:
        # Lire le CSV et générer un aperçu sous forme de tableau HTML
        file_path = os.path.join(files_path, selected_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Nombre de lignes par page
            rows_per_page = 8

            # Calculer le nombre total de pages
            total_rows = len(df)
            total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)

            # Découper le DataFrame pour la page actuelle
            start_row = (current_page - 1) * rows_per_page
            end_row = start_row + rows_per_page
            page_data = df.iloc[start_row:end_row]

            # Convertir les données de la page en tableau HTML
            file_preview = page_data.to_html(classes="preview-table", index=False)

            # Passer les informations de pagination au template
            return render_template('dataset_view.html', files=files, file_preview=file_preview,
                                   selected_file=selected_file, current_page=current_page, total_pages=total_pages)

    return render_template('dataset_view.html', files=files, file_preview=file_preview, selected_file=selected_file)

#________________#MACHINE LEARNING_________________________

# Route pour charger le modèle et faire des prédictions

# Chemin du modèle et du fichier CSV
MODEL_PATH = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl'
CSV_FOLDER = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/'
 #Route pour charger le modèle et générer des prédictions

# Route pour charger le modèle et générer des prédictions
from sklearn.preprocessing import LabelEncoder

@app.route('/load_model_and_predict', methods=['GET', 'POST'])
def load_model_and_predict():
    # Récupérer le chemin du fichier CSV à partir de la session
    csv_path = session.get('csv_path')

    if not csv_path:
        flash("Aucun fichier CSV disponible", "danger")
        return redirect(url_for('smart_parking'))

    # Charger les données CSV
    try:
        new_data = pd.read_csv(csv_path)
    except Exception as e:
        flash(f"Erreur lors du chargement du fichier CSV: {str(e)}", "danger")
        return redirect(url_for('smart_parking'))

    # Charger le modèle
    try:
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé avec succès")
    except Exception as e:
        flash(f"Erreur lors du chargement du modèle: {str(e)}", "danger")
        return redirect(url_for('smart_parking'))

    # Prétraitement des données
    if 'departure_time' in new_data.columns and 'arrival_time' in new_data.columns:
        new_data['departure_time'] = pd.to_datetime(new_data['departure_time'])
        new_data['arrival_time'] = pd.to_datetime(new_data['arrival_time'])
        new_data['flight_duration'] = (new_data['arrival_time'] - new_data['departure_time']).dt.total_seconds() / 60
    else:
        new_data['flight_duration'] = np.nan

    # Ajouter la colonne 'live' si elle n'existe pas
    if 'live' not in new_data.columns:
        new_data['live'] = 0

    # Encoder la colonne 'live' si elle contient des chaînes de caractères
    if new_data['live'].dtype == 'object':
        le_live = LabelEncoder()
        new_data['live'] = le_live.fit_transform(new_data['live'].astype(str))

    # Vérifier que la colonne 'live' est de type numérique
    if new_data['live'].dtype != 'int' and new_data['live'].dtype != 'float':
        new_data['live'] = new_data['live'].astype(int)

    # Encoder les autres colonnes
    le_flight = LabelEncoder()
    le_airline = LabelEncoder()
    le_flight_status = LabelEncoder()
    le_departure_airport = LabelEncoder()
    le_arrival_airport = LabelEncoder()

    new_data['flight'] = le_flight.fit_transform(new_data['flight'].astype(str))
    new_data['airline'] = le_airline.fit_transform(new_data['airline'].astype(str))
    new_data['flight_status'] = le_flight_status.fit_transform(new_data['flight_status'].astype(str))
    new_data['departure_airport'] = le_departure_airport.fit_transform(new_data['departure_airport'].astype(str))
    new_data['arrival_airport'] = le_arrival_airport.fit_transform(new_data['arrival_airport'].astype(str))

    # Préparer les données pour la prédiction
    X_new = new_data[['flight_status', 'airline', 'flight', 'live', 'departure_airport', 'arrival_airport', 'flight_duration']]

    # Prédire avec le modèle
    predictions = model.predict(X_new)

    # Ajouter les prédictions
    new_data['predicted_needs_parking'] = np.where(
        predictions == 1,
        "L'avion nécessite un stationnement sur le tarmac",
        "L' avion ne nécessite pas de stationnement sur le tarmac"
    )

    # Décoder les compagnies aériennes et autres colonnes
    new_data['airline'] = le_airline.inverse_transform(new_data['airline'])
    new_data['flight_status'] = le_flight_status.inverse_transform(new_data['flight_status'])

    # Mettre à jour 'needs_parking' en fonction des prédictions et du flight_status
    def update_parking_based_on_status(row):
        if row['flight_status'] == 'landed':  # Le statut est maintenant sous forme de chaîne, pas encodé
            return "Besoin de stationnement sur le tarmac"
        elif row['flight_status'] == 'scheduled':
            return "Vol planifié, aucun besoin immédiat"
        elif row['flight_status'] == 'active':
            return "En vol, aucun besoin immédiat"
        return row['predicted_needs_parking']

    new_data['needs_parking'] = new_data.apply(update_parking_based_on_status, axis=1)

    # Sélectionner les colonnes pour l'affichage dans la page HTML
    results_df = new_data[['flight_date', 'flight', 'airline', 'flight_status', 'needs_parking', 'predicted_needs_parking']]

    # Pagination des résultats
    page = request.args.get('page', 1, type=int)
    per_page = 6
    total_pages = len(results_df) // per_page + (1 if len(results_df) % per_page > 0 else 0)

    # Calculer les indices de la page
    start = (page - 1) * per_page
    end = start + per_page
    page_data = results_df.iloc[start:end]

    # Mettre à jour la session avec les prédictions
    session['predictions_data'] = page_data.to_dict(orient='records')

    # Sauvegarder le fichier de prédictions
    save_path = os.path.join(CSV_FOLDER, 'predictions_data.csv')
    results_df.to_csv(save_path, index=False)
    print(f"Fichier de prédictions sauvegardé avec succès à : {save_path}")

    return render_template('prediction.html', page_data=page_data.to_dict(orient='records'), page=page, total_pages=total_pages)

#______________VISUALISATION DES DATASETS ET ANALYSES______________

# Chemin vers le dossier contenant le fichier CSV
app.config['UPLOAD_FOLDER'] = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'
optim_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Vols_optimisés.csv')
pred_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_data.csv')

# 1. Route pour afficher la page des visualisations

# _______Route pour afficher les visualisations_______
@app.route('/visualizations', methods=['GET', 'POST'])
def show_visualizations():
    # Taille de page pour la pagination (5 lignes par page)
    rows_per_page = 5

    # Récupérer les paramètres depuis l'URL

    show_data = request.args.get('show_data', False, type=bool)  # Récupérer show_data

    # Récupérer les numéros de page pour la pagination
    page_pred = int(request.args.get('page_pred', 1))  # Page pour le dataset de prédiction
    page_optim = int(request.args.get('page_optim', 1))  # Page pour le dataset d'optimisation

    # Vérifie si le dataset de prédiction existe
    if 'prediction_df' in session:
        prediction_df = pd.DataFrame(session['prediction_df'])
        total_pages_pred = (len(prediction_df) - 1) // rows_per_page + 1  # Calcul du nombre total de pages
        start_pred = (page_pred - 1) * rows_per_page
        end_pred = start_pred + rows_per_page
        prediction_data = prediction_df.iloc[start_pred:end_pred].to_dict(orient='records')


    else:
        prediction_data = []
        total_pages_pred = 1  # Si aucun dataset de prédiction n'est chargé, on a une seule page


    # Gestion du dataset d'optimisation
    if 'optim_df' in session:
        optim_df = pd.DataFrame(session['optim_df'])
        print(f"Optim data in session: {optim_df.head()}")  # Log pour vérifier les données
        total_pages_optim = (len(optim_df) - 1) // rows_per_page + 1
        start_optim = (page_optim - 1) * rows_per_page
        end_optim = start_optim + rows_per_page
        optim_data = optim_df.iloc[start_optim:end_optim].to_dict(orient='records')
    else:
        optim_data = []
        total_pages_optim = 1  # Si aucun dataset d'optimisation n'est chargé, on a une seule page
        print("Optim data not found in session.")

    # Passer les variables nécessaires au template
    return render_template(
        'visualizations.html',
        prediction_data=prediction_data if show_data else [],  # Afficher les données si show_data est True
        total_pages_pred=total_pages_pred,
        current_page_pred=page_pred,
        optim_data=optim_data if show_data else [],  # Afficher les données si show_data est True
        total_pages_optim=total_pages_optim,
        current_page_optim=page_optim,


    )


# Route pour uploader le dataset de prédiction
UPLOAD_FOLDER = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'
ALLOWED_EXTENSIONS = {'csv'}

# Fonction pour vérifier si le fichier est autorisé (CSV)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ______Route pour uploader le dataset de prédiction_______
@app.route('/upload_prediction_dataset', methods=['POST'])
def upload_prediction_dataset():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("Aucun fichier sélectionné.", "error")
        session['prediction_uploaded'] = False
        return redirect(url_for('show_visualizations'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        try:
            # Charger le dataset dans la session
            prediction_df = pd.read_csv(file_path)
            session['prediction_df'] = prediction_df.to_dict(orient='records')
            session['prediction_uploaded'] = True
            flash("Dataset téléchargé avec succès.", "success")
        except Exception as e:
            flash(f"Erreur de lecture du fichier : {e}", "error")
            session['prediction_uploaded'] = False
    else:
        flash("Format de fichier non valide. Veuillez uploader un fichier CSV.", "error")
        session['prediction_uploaded'] = False

    return redirect(url_for('show_visualizations'))

#______ Route pour uploader le dataset d'optimisation______
# Route pour uploader le dataset d'optimisation
@app.route('/upload_optim_dataset', methods=['POST'])
def upload_optim_dataset():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("Aucun fichier sélectionné.", "error")
        session['optim_uploaded'] = False
        return redirect(url_for('show_visualizations'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Vérification du fichier téléchargé
        print(f"Fichier téléchargé : {file_path}")  # Log du chemin du fichier
        if not os.path.exists(file_path):
            flash(f"Le fichier n'a pas été trouvé à l'emplacement : {file_path}", "error")
            session['optim_uploaded'] = False
            return redirect(url_for('show_visualizations'))

        try:
            optim_df = pd.read_csv(file_path)
            # Log des données avant de les stocker dans la session
            print(f"Data dans le CSV avant stockage : {optim_df.head()}")  # Affiche les premières lignes du CSV

            session['optim_df'] = optim_df.to_dict(orient='records')
            session['optim_uploaded'] = True
            flash("Dataset téléchargé avec succès.", "success")
        except Exception as e:
            flash(f"Erreur de lecture du fichier : {e}", "error")
            session['optim_uploaded'] = False
    else:
        flash("Format de fichier non valide. Veuillez uploader un fichier CSV.", "error")
        session['optim_uploaded'] = False

    # Ajoute un log ici pour vérifier que les données sont stockées dans la session
    print(f"Data dans la session après upload : {session.get('optim_df', 'Aucune donnée dans la session')}")

    return redirect(url_for('show_visualizations'))




  #_______________OPTIMISATION DES DONEES__________________________

# Route pour générer l'optimisation et sauvegarder les résultats
@app.route('/generer_optimisation', methods=['GET'])
def generer_optimisation():
    try:
        # Charger le dataset
        dataset_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/predictions_data.csv'
        data = pd.read_csv(dataset_path)

        # Vérification des colonnes nécessaires
        required_columns = ['flight_date', 'flight', 'airline', 'flight_status', 'needs_parking', 'predicted_needs_parking']
        if not all(col in data.columns for col in required_columns):
            return "Le dataset ne contient pas toutes les colonnes nécessaires.", 400

        # Extraire l'heure d'arrivée depuis 'flight_date'
        try:
            data['flight_date'] = pd.to_datetime(data['flight_date'], errors='coerce')  # Conversion en datetime
            data['arrival_hour'] = data['flight_date'].dt.hour  # Extraire l'heure de la colonne 'flight_date'
        except Exception as e:
            return f"Erreur lors de l'extraction des heures : {str(e)}", 500

        # Créer la colonne 'can_park' en fonction de 'predicted_needs_parking'
        def generate_can_park(predicted_needs_parking):
            if "nécessite" in predicted_needs_parking:
                return 1  # L'avion nécessite un stationnement
            else:
                return 0  # L'avion ne nécessite pas de stationnement

        data['can_park'] = data['predicted_needs_parking'].apply(generate_can_park)

        # Fonction pour générer des alertes
        def generate_alert(row):
            peak_hours = [8, 11, 18, 19, 20, 23]
            if row['predicted_needs_parking'] == "L' avion ne nécessite pas de stationnement sur le tarmac":
                if row['arrival_hour'] in peak_hours:
                    return f"ALERTE : L'avion de la compagnie {row['airline']} ne peut pas stationner en raison des heures de pointe."
                else:
                    return f"L'avion de la compagnie {row['airline']} n'est pas autorisé à stationner sur le tarmac."
            elif row['predicted_needs_parking'] == "L'avion nécessite un stationnement sur le tarmac":
                if row['can_park'] == 1:
                    if row['arrival_hour'] in peak_hours:
                        return f"ALERTE : L'atterrissage de l'avion de la compagnie {row['airline']} aura lieu pendant les heures de pointe. Un aménagement immédiat doit être fait."
                    else:
                        return f"L'avion de la compagnie {row['airline']} est autorisé à stationner sur le tarmac."
                else:
                    return f"ALERTE : L'avion de la compagnie {row['airline']} ne peut pas stationner sur le tarmac pour le moment."
            return "Données non reconnues pour générer une alerte."

        # Appliquer la fonction pour générer des alertes
        data['alert'] = data.apply(generate_alert, axis=1)

# Réorganiser le DataFrame pour ne conserver que les colonnes nécessaires
        data = data[['flight_date', 'airline', 'arrival_hour', 'predicted_needs_parking', 'can_park', 'alert']]

        # Sauvegarde des résultats optimisés dans un nouveau fichier CSV
        save_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Vols_optimisés.csv'
        data.to_csv(save_path, index=False)

        # Pagination des résultats
        page = int(request.args.get('page', 1))
        per_page = 8
        total_pages = -(-len(data) // per_page)  # Calcul du nombre de pages
        start = (page - 1) * per_page
        end = start + per_page

        paginated_data = data.iloc[start:end].to_dict(orient='records')

        # Sauvegarde des résultats paginés dans la session
        session['optimized_data'] = paginated_data

        # Rendre le template avec les données paginées
        return render_template('optim.html', optimized_data=paginated_data, page=page, total_pages=total_pages)

    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
