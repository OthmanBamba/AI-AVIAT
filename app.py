from flask import Flask, render_template, request, redirect, url_for, session,flash
import pandas as pd
import pickle
import os
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Charger le modèle
model_path = "/Users/benothmane/Desktop/flight_dashboard/models/random_forest_model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
# Spécifiez le chemin du dossier où les fichiers sont sauvegardés
UPLOAD_FOLDER = '/Users/benothmane/Desktop/flight_dashboard/upload/folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route de connexion
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "password":  # Exemple d'authentification
            session['logged_in'] = True
            return redirect(url_for('home'))  # Rediriger vers 'home' après une connexion réussie
    return render_template('login.html')  # Afficher la page de connexion

# Route de déconnexion
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Route de la page d'accueil
@app.route('/home', methods=['GET'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

# Route pour afficher et analyser le fichier CSV
@app.route('/smart_parking', methods=['GET', 'POST'])
def smart_parking():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    csv_data = []
    csv_path = None
    page = request.args.get('page', 1, type=int)
    rows_per_page = 8  # Nombre de lignes par page

    if request.method == 'POST':
        # Vérifiez si le formulaire est pour télécharger un CSV
        if 'dataset' in request.files:
            csv_file = request.files['dataset']
            if csv_file:
                # Lire le fichier CSV
                data = pd.read_csv(csv_file)

                # Sauvegarder le fichier CSV pour l'utiliser plus tard
                csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'flights.csv')
                data.to_csv(csv_path, index=False)

                # Stocker les données en session pour la pagination
                session['csv_data'] = data.to_dict(orient='records')
                session['csv_length'] = len(data)  # Nombre total de lignes

                flash('Fichier CSV téléchargé et sauvegardé avec succès !')

        # Vérifiez si le formulaire est pour sauvegarder le CSV (si vous avez un bouton pour cela)
        if 'csv_data' in request.form:
            csv_data_to_save = request.form['csv_data']
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'saved_flights.csv')
            with open(save_path, 'w') as f:
                f.write(csv_data_to_save)
            flash('Fichier CSV sauvegardé avec succès !')

    # Pagination
    if 'csv_data' in session:
        start = (page - 1) * rows_per_page
        end = start + rows_per_page
        csv_data = session['csv_data'][start:end]
        total_pages = (session['csv_length'] // rows_per_page) + (1 if session['csv_length'] % rows_per_page > 0 else 0)
    else:
        total_pages = 0

    return render_template('smart_parking.html', csv_data=csv_data, page=page, total_pages=total_pages, csv_path=csv_path)


# Route pour télécharger le modèle
@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        flash("Aucun fichier sélectionné", "danger")
        return redirect(url_for('smart_parking'))

    model_file = request.files['model']
    if model_file.filename == '':
        flash("Aucun fichier sélectionné", "danger")
        return redirect(url_for('smart_parking'))

    if allowed_file(model_file.filename):
        model_path = "/Users/benothmane/Desktop/flight_dashboard/upload/folder/" + model_file.filename
        model_file.save(model_path)  # Enregistrer le modèle sur le serveur
        flash("Modèle téléchargé avec succès !", "success")
        return redirect(url_for('smart_parking'))

    flash("Format de fichier non autorisé", "danger")
    return redirect(url_for('smart_parking'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pkl'  # Seul les fichiers .pkl




# Route pour générer la prédiction
@app.route('/generate_prediction', methods=['POST'])
def generate_prediction():
    if 'csv_data' not in session:
        flash("Aucun fichier CSV chargé pour les prédictions.", "danger")
        return redirect(url_for('smart_parking'))

    csv_path = "/Users/benothmane/Desktop/flight_dashboard/upload/folder/flights.csv"

    # Lire le fichier CSV pour les prédictions
    data = pd.read_csv(csv_path)

    # Assurez-vous que les colonnes requises existent
    required_columns = ['airline', 'flight', 'aircraft', 'departure_airport', 'arrival_airport']
    if not all(col in data.columns for col in required_columns):
        flash("Le fichier CSV ne contient pas toutes les colonnes nécessaires.", "danger")
        return redirect(url_for('smart_parking'))

    # Faire la prédiction avec le modèle
    predictions = model.predict(data[required_columns])
    data['Needs Parking'] = predictions

    # Convertir les données en HTML pour l'afficher dans prediction.html
    table_html = data.to_html(classes='table table-striped')

    # Compter les valeurs de stationnement
    need_parking = data['Needs Parking'].value_counts().to_dict()

@app.route('/view_datasets')
def view_datasets():
    # Lister les fichiers CSV dans le dossier
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
    return render_template('view_datasets.html', files=files)

@app.route('/home', methods=['GET', 'POST'])
def home_view():  # Renommez la fonction ici
    csv_files = []  # Liste pour stocker les fichiers CSV

    # Lister les fichiers CSV dans le dossier
    files_path = app.config['UPLOAD_FOLDER']
    if os.path.exists(files_path):
        csv_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]

    # Vérifier quelle section afficher
    section = request.args.get('section', 'accueil')  # Par défaut, afficher l'accueil

    return render_template('home.html', csv_files=csv_files, section=section)
    

if __name__ == '__main__':
    app.run(debug=True)
