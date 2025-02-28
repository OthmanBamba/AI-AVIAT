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

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Charger le modèle pré-entraîné
model_path = "/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl"
model = joblib.load(model_path)  # Utilisez joblib.load pour charger le modèle
# Configuration du dossier de sauvegarde des fichiers
app.config['UPLOAD_FOLDER'] = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'


# Variables pour stocker les datasets
main_dataset = None
predict_dataset = None

#LOGIN
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
#PAGE D ACCUEIL
# Route de la page d'accueil
@app.route('/home', methods=['GET'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')



#AFFICHAGE  DE DONNEES sur Smart_parking

# Route pour afficher et analyser le fichier CSV principal
@app.route('/smart_parking', methods=['GET', 'POST'])
def smart_parking():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    csv_data = []
    csv_path = None
    page = request.args.get('page', 1, type=int)
    rows_per_page = 6

    if request.method == 'POST' and 'dataset' in request.files:
        csv_file = request.files['dataset']
        if csv_file:
            # Sauvegarde du fichier CSV
            csv_filename = 'flights.csv'
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
            csv_file.save(csv_path)
            session['csv_path'] = csv_path
            data = pd.read_csv(csv_path)
            session['csv_data'] = data.to_dict(orient='records')
            session['csv_length'] = len(data)
            flash('Fichier CSV téléchargé avec succès !')

     # Gestion de la pagination
    if 'csv_data' in session:
        start = (page - 1) * rows_per_page
        end = start + rows_per_page
        csv_data = session['csv_data'][start:end]
        total_pages = (session['csv_length'] // rows_per_page) + (1 if session['csv_length'] % rows_per_page > 0 else 0)
    else:
        total_pages = 0

    # Transmettre explicitement la variable page
    return render_template(
        'smart_parking.html',
        csv_data=csv_data,
        page=page,  # Variable page passée ici
        total_pages=total_pages,
        csv_path=session.get('csv_path', None)
    )


# Route pour sauvegarder les données CSV
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

    # Route de retour à la page d'accueil
    @app.route('/home', methods=['GET', 'POST'])
    def home_view():
        csv_files = []
        files_path = app.config['UPLOAD_FOLDER']
        if os.path.exists(files_path):
            csv_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
        section = request.args.get('section', 'accueil')
        return render_template('home.html', csv_files=csv_files, section=section)

#DATASET TELECHARGES

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

    if selected_file:
        # Lire le CSV et générer un aperçu sous forme de tableau HTML
        file_path = os.path.join(files_path, selected_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convertir le DataFrame en table HTML
            file_preview = df.head(10).to_html(classes="preview-table", index=False)

    return render_template('dataset_view.html', files=files, file_preview=file_preview, selected_file=selected_file)

        #MACHINE LEARNING



 # Route pour charger le modèle et faire des prédictions

   #MACHINE LEARNING
# Chemin du modèle et du fichier CSV
MODEL_PATH = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl'
CSV_FOLDER = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/'
 #Route pour charger le modèle et générer des prédictions

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
    model_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Train_model2.pkl'
    try:
        model = joblib.load(model_path)
    except Exception as e:
        flash(f"Erreur lors du chargement du modèle: {str(e)}", "danger")
        return redirect(url_for('smart_parking'))

    # Effectuer les prédictions
    if 'departure_time' in new_data.columns and 'arrival_time' in new_data.columns:
        new_data['departure_time'] = pd.to_datetime(new_data['departure_time'])
        new_data['arrival_time'] = pd.to_datetime(new_data['arrival_time'])
        new_data['flight_duration'] = (new_data['arrival_time'] - new_data['departure_time']).dt.total_seconds() / 60
    else:
        new_data['flight_duration'] = np.nan

    if 'live' not in new_data.columns:
        new_data['live'] = 0

    if new_data['live'].dtype == 'object':
        le_live = LabelEncoder()
        new_data['live'] = le_live.fit_transform(new_data['live'].astype(str))

    # Encodage des autres colonnes
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

    # Ajouter les prédictions à la colonne 'predicted_needs_parking'
    new_data['predicted_needs_parking'] = np.where(
        predictions == 1,
        "L'avion nécessite un stationnement sur le tarmac",
        "L' avion ne nécessite pas de stationnement sur le tarmac"
    )

    # Mettre à jour 'needs_parking' en fonction des prédictions et du flight_status
    def update_parking_based_on_status(row):
        if row['flight_status'] == le_flight_status.transform(['landed'])[0]:
            return "Besoin de stationnement sur le tarmac"
        elif row['flight_status'] == le_flight_status.transform(['scheduled'])[0]:
            return "Vol planifié, aucun besoin immédiat"
        elif row['flight_status'] == le_flight_status.transform(['active'])[0]:
            return "En vol, aucun besoin immédiat"
        return row['predicted_needs_parking']

    new_data['needs_parking'] = new_data.apply(update_parking_based_on_status, axis=1)

    # Décoder 'flight_status' pour afficher les valeurs originales
    new_data['flight_status'] = le_flight_status.inverse_transform(new_data['flight_status'])

    # Décoder 'airline' pour afficher les noms originaux des compagnies aériennes
    new_data['airline'] = le_airline.inverse_transform(new_data['airline'])

    # Préparer le DataFrame final
    result = new_data[['flight_date', 'flight', 'airline', 'flight_status', 'needs_parking', 'predicted_needs_parking']]

    # Pagination des résultats
    page = request.args.get('page', 1, type=int)
    per_page = 6
    total_pages = len(result) // per_page + (1 if len(result) % per_page > 0 else 0)

    # Calculer les indices de la page
    start = (page - 1) * per_page
    end = start + per_page
    page_data = result.iloc[start:end]

    # Mettre à jour les données de session
    session['predictions_data'] = page_data.to_dict(orient='records')

    # Sauvegarder tout le dataset de prédictions (au lieu de la page actuelle)
    if 'csv_saved' not in session:  # Sauvegarder une seule fois
        try:
            save_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/predictions_data.csv'
            result.to_csv(save_path, index=False)  # Sauvegarder tout le dataset
            session['csv_saved'] = True  # Marquer comme sauvegardé
            flash("Les prédictions ont été générées et stockées dans la session. Le fichier CSV a été sauvegardé.", "success")
        except Exception as e:
            flash(f"Erreur lors de la sauvegarde du fichier CSV : {e}", "danger")

    # Retourner les données avec la pagination au template
    return render_template('prediction.html', page_data=page_data.to_dict(orient='records'), page=page, total_pages=total_pages)


#VISUALISATION DES DATASETS ET ANALYSES

# Chemin vers le dossier contenant le fichier CSV
app.config['UPLOAD_FOLDER'] = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'
optim_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Vols_optimisés.csv.csv')
pred_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_data.csv')

# 1. Route pour afficher la page des visualisations
# Route pour afficher les visualisations
@app.route('/visualizations', methods=['GET', 'POST'])
def show_visualizations():
    # Taille de page pour la pagination (5 lignes par page)
    rows_per_page = 5
    show_graphs = request.args.get('show_graphs', False)

    # Récupérer les numéros de page depuis les paramètres GET (par défaut la page 1)
    page_pred = int(request.args.get('page_pred', 1))  # Page pour le dataset de prédiction
    page_optim = int(request.args.get('page_optim', 1))  # Page pour le dataset d'optimisation

    # Gestion du dataset de prédiction
    if 'prediction_df' in session:
        prediction_df = pd.DataFrame(session['prediction_df'])
        total_pages_pred = (len(prediction_df) - 1) // rows_per_page + 1  # Calcul du nombre total de pages
        start_pred = (page_pred - 1) * rows_per_page
        end_pred = start_pred + rows_per_page
        prediction_data = prediction_df.iloc[start_pred:end_pred].to_dict(orient='records')
    else:
        prediction_data = []
        total_pages_pred = 1  # Si aucun dataset de prédiction n'est chargé, on a une seule page
# Générer les graphiques pour le dataset de prédiction
    if prediction_data:
        pred_pie_chart_1, pred_pie_chart_2, pred_histogram = generate_prediction_charts(prediction_df)


    # Gestion du dataset d'optimisation
    if 'optim_df' in session:
        optim_df = pd.DataFrame(session['optim_df'])
        print("Données du dataset d'optimisation : ", optim_df.head())  # Debug
        total_pages_optim = (len(optim_df) - 1) // rows_per_page + 1
        start_optim = (page_optim - 1) * rows_per_page
        end_optim = start_optim + rows_per_page
        optim_data = optim_df.iloc[start_optim:end_optim].to_dict(orient='records')
    else:
        optim_data = []
        total_pages_optim = 1
        

    # Passer toutes les données au template
    return render_template(
        'visualizations.html',
        prediction_data=prediction_data,
        total_pages_pred=total_pages_pred,
        current_page_pred=page_pred,
        optim_data=optim_data,  # Données pour le dataset d'optimisation
        total_pages_optim=total_pages_optim,  # Nombre total de pages pour l'optimisation
        current_page_optim=page_optim, # Page actuelle pour l'optimisation
        pred_pie_chart_1=pred_pie_chart_1,  # Premier graphique Camembert
        pred_pie_chart_2=pred_pie_chart_2,  # Deuxième graphique Camembert
        pred_histogram=pred_histogram

    )
# Fonction pour générer les graphiques pour le dataset de prédiction
def generate_prediction_charts(df):
    # 1. Diagramme en camembert pour 'predicted_needs_parking'
    counts_1 = df['predicted_needs_parking'].value_counts()
    fig_pie_1, ax_pie_1 = plt.subplots(figsize=(8, 8))
    ax_pie_1.pie(counts_1, labels=counts_1.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax_pie_1.set_title("Répartition des Besoins de Stationnement Prédit")
    ax_pie_1.axis('equal')  # Assurer que le graphique est un cercle

    # Convertir le graphique en image encodée en base64
    img_pie_1 = BytesIO()
    fig_pie_1.savefig(img_pie_1, format='png')
    img_pie_1.seek(0)
    pred_pie_chart_1_url = base64.b64encode(img_pie_1.getvalue()).decode('utf-8')
    plt.close(fig_pie_1)

    # 2. Diagramme en camembert pour un autre champ, par exemple 'flight_status'
    counts_2 = df['flight_status'].value_counts()
    fig_pie_2, ax_pie_2 = plt.subplots(figsize=(8, 8))
    ax_pie_2.pie(counts_2, labels=counts_2.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax_pie_2.set_title("Répartition des Vols selon leur Statut")
    ax_pie_2.axis('equal')  # Assurer que le graphique est un cercle

    # Convertir le graphique en image encodée en base64
    img_pie_2 = BytesIO()
    fig_pie_2.savefig(img_pie_2, format='png')
    img_pie_2.seek(0)
    pred_pie_chart_2_url = base64.b64encode(img_pie_2.getvalue()).decode('utf-8')
    plt.close(fig_pie_2)

    # 3. Histogramme pour 'predicted_needs_parking'
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    counts_1.plot(kind='bar', color=['#66b3ff', '#ff6666'], ax=ax_hist)
    ax_hist.set_title("Histogramme des Besoins de Stationnement Prédit")
    ax_hist.set_xlabel('Besoins de Stationnement Prédit')
    ax_hist.set_ylabel('Nombre de Vols')
    ax_hist.set_xticklabels(counts_1.index, rotation=0)
    plt.tight_layout()

    # Convertir l'histogramme en image encodée en base64
    img_hist = BytesIO()
    fig_hist.savefig(img_hist, format='png')
    img_hist.seek(0)
    pred_histogram_url = base64.b64encode(img_hist.getvalue()).decode('utf-8')
    plt.close(fig_hist)

    return pred_pie_chart_1_url, pred_pie_chart_2_url, pred_histogram_url

# Route pour uploader le dataset de prédiction
UPLOAD_FOLDER = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder'  # Répertoire où les fichiers seront enregistrés
ALLOWED_EXTENSIONS = {'csv'}

# Fonction pour vérifier si le fichier est autorisé (CSV)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route pour uploader le dataset de prédiction
@app.route('/upload_prediction_dataset', methods=['POST'])
def upload_prediction_dataset():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)  # Enregistrer le fichier

        # Charger les données du fichier CSV et les stocker dans la session
        try:
            prediction_df = pd.read_csv(file_path)
            session['prediction_df'] = prediction_df.to_dict(orient='records')  # Stocker les données dans la session
            flash("Dataset de prédiction téléchargé avec succès.", "success")
        except Exception as e:
            flash(f"Erreur lors du chargement du fichier CSV : {e}", "error")

        return redirect(url_for('show_visualizations'))  # Rediriger vers la page des visualisations
    else:
        flash("Veuillez télécharger un fichier CSV valide.", "error")
        return redirect(request.url)

# Route pour uploader le dataset d'optimisation
@app.route('/upload_optim_dataset', methods=['POST'])
def upload_optim_dataset():
    if 'file' not in request.files:
        return "Aucun fichier sélectionné", 400

    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné", 400

    if file:
        try:
            # Charger le fichier en DataFrame
            optim_df = pd.read_csv(file)

            # Vérifier que le fichier contient des données valides (facultatif)
            if optim_df.empty:
                return "Le fichier est vide ou invalide", 400

            # Stocker dans la session
            session['optim_df'] = optim_df.to_dict(orient='records')

            print("Dataset d'optimisation chargé avec succès")  # Debug
            return redirect(url_for('show_visualizations'))
        except Exception as e:
            return f"Erreur lors du traitement du fichier : {e}", 500

  #OPTIMISATION DES DONEES

# Route pour générer l'optimisation et sauvegarder les résultats
# Liste des heures de pointe
peak_hours = [7, 8, 9, 18, 19, 20]

# Fonction pour extraire l'heure à partir du 'flight_status'
def extract_hour(flight_status):
    try:
        # Assurez-vous que 'flight_status' est bien un nombre et contient une valeur valide
        return int(str(flight_status)[-2:])  # On prend les 2 derniers chiffres comme heure
    except ValueError:
        return None  # Retourner None si la conversion échoue (valeur invalide)

# Fonction pour générer les alertes
def generate_alert(row):
    if row['predicted_needs_parking'] == "L' avion ne nécessite pas de stationnement sur le tarmac":
        if row['arrival_hour'] in peak_hours:
            return f"ALERTE : L'avion de la compagnie {row['airline']} ne nécessite pas de stationnement mais se trouve pendant les heures de pointe."
        else:
            return f"L'avion de la compagnie {row['airline']} ne nécessite pas de stationnement."

    elif row['predicted_needs_parking'] == "L'avion nécessite un stationnement sur le tarmac":
        if row['can_park'] == 1:
            if row['arrival_hour'] in peak_hours:
                return f"L'avion de la compagnie {row['airline']} est autorisé à stationner sur le tarmac pendant les heures de pointe."
            else:
                return f"L'avion de la compagnie {row['airline']} est autorisé à stationner sur le tarmac."
        else:
            return f"ALERTE : L'avion de la compagnie {row['airline']} nécessite un stationnement, mais il ne peut pas stationner."

    return "Données non reconnues pour générer une alerte."

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

        # Extraction de l'heure d'arrivée depuis 'flight_status'
        def extract_hour(flight_status):
            # Extraire l'heure depuis la colonne 'flight_status', en supposant que le format soit HH
            try:
                return int(flight_status.split(':')[0])  # extrait l'heure de l'heure de vol
            except Exception as e:
                return None

        # Appliquer l'extraction de l'heure d'arrivée
        data['arrival_hour'] = data['flight_status'].apply(extract_hour)

        # Créer la colonne 'can_park' en fonction de 'predicted_needs_parking'
        def generate_can_park(predicted_needs_parking):
            if "nécessite" in predicted_needs_parking:
                return 1  # L'avion nécessite un stationnement
            else:
                return 0  # L'avion ne nécessite pas de stationnement

        data['can_park'] = data['predicted_needs_parking'].apply(generate_can_park)

        # Fonction pour générer des alertes
        def generate_alert(row):
            peak_hours = [8, 11, 18, 19, 20,23]
            if row['predicted_needs_parking'] == "L' avion ne nécessite pas de stationnement sur le tarmac":
                if row['arrival_hour'] in peak_hours:
                    return f"ALERTE :L'avion de la compagnie {row['airline']} ne peut pas stationner en raison des heures de pointe."
                else:
                    return f"L'avion de la compagnie {row['airline']} n'est pas autorisé à stationner sur le tarmac."
            elif row['predicted_needs_parking'] == "L'avion nécessite un stationnement sur le tarmac":
                if row['can_park'] == 1:
                    if row['arrival_hour'] in peak_hours:
                        return f"Latterissage de l'avion de la compagnie {row['airline']} aura lieu  Pendant les heures de pointe, un aménagement immédiat doit être fait."
                    else:
                        return f"L'avion de la compagnie {row['airline']} est autorisé à stationner sur le tarmac."
                else:
                    return f"ALERTE : L'avion de la compagnie {row['airline']} ne peut pas stationner sur le tarmac pour le moment "
            return "Données non reconnues pour générer une alerte."

        # Appliquer la fonction pour générer des alertes
        data['alert'] = data.apply(generate_alert, axis=1)

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
