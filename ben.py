from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Pour utiliser flash
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Vérifie si le fichier a une extension valide
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Page principale

@app.route('/upload_prediction_dataset', methods=['POST'])
def upload_prediction_dataset():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash("Aucun fichier sélectionné.", "error")
        return redirect(url_for('upload_prediction_dataset'))

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
            return redirect(url_for('show_dataset'))  # Rediriger vers la page qui affiche le dataset
        except Exception as e:
            flash(f"Erreur de lecture du fichier : {e}", "error")
            session['prediction_uploaded'] = False
            return redirect(url_for('upload_prediction_dataset'))
    else:
        flash("Format de fichier non valide. Veuillez uploader un fichier CSV.", "error")
        return redirect(url_for('upload_prediction_dataset'))

@app.route('/show_dataset', methods=['GET'])
def show_dataset():
    if 'prediction_df' in session:
        df = pd.DataFrame(session['prediction_df'])  # Charger les données depuis la session
        return render_template('show_dataset.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    else:
        flash("Aucun dataset téléchargé.", "error")
        return redirect(url_for('upload_prediction_dataset'))

if __name__ == '__main__':
    app.run(debug=True)
