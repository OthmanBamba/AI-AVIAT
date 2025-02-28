from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np  # Importation manquante

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Route pour l'authentification
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        # Simuler une connexion simple (à remplacer par une vraie validation)
        if username == "admin" and password == "password":  # Valeurs d'exemple
            session["user"] = username
            return redirect(url_for("main"))
        else:
            error = "Identifiant ou mot de passe incorrect"
            return render_template("index.html", error=error)
    return render_template("index.html")

# Route de déconnexion
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

@app.route("/main")
def main():
    if "user" not in session:
        return redirect(url_for("index"))

    try:
        DATA_PATH = "/Users/benothmane/Desktop/flight_dashboard/static/dataset.csv"
        dataset = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return "Erreur : Fichier non trouvé. Vérifiez le chemin du dataset."

    preview_data = dataset.head(10).to_html(classes="table table-striped", index=False)
    return render_template("main.html", preview_data=preview_data)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user" not in session:
        return redirect(url_for("index"))

    try:
        # Charger le modèle
        MODEL_PATH = "/Users/benothmane/Desktop/flight_dashboard/models/random_forest_model.pkl"
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return "Erreur : Fichier du modèle non trouvé. Vérifiez le chemin du modèle."

    try:
        DATA_PATH = "/Users/benothmane/Desktop/flight_dashboard/static/dataset.csv"
        dataset = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return "Erreur : Fichier non trouvé. Vérifiez le chemin du dataset."

    # Prétraitement pour correspondre aux besoins du modèle
    dataset['departure_time'] = pd.to_datetime(dataset['departure_time'])
    dataset['arrival_time'] = pd.to_datetime(dataset['arrival_time'])
    dataset['departure_hour'] = dataset['departure_time'].dt.hour
    dataset['arrival_hour'] = dataset['arrival_time'].dt.hour

    # Prédictions
    X = dataset[['departure_hour', 'arrival_hour']]  # Simplifié pour exemple
    dataset['predicted_parking'] = model.predict(X)

    # Visualisations
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(dataset['predicted_parking'], bins=2, color='skyblue')
    axs[0].set_title("Histogramme des besoins de stationnement")

    counts = dataset['predicted_parking'].value_counts()
    axs[1].pie(counts, labels=['Pas de stationnement', 'Stationnement nécessaire'], autopct='%1.1f%%', startangle=140)
    axs[1].set_title("Distribution des besoins de stationnement")

    # Sauvegarde des visualisations dans un fichier temporaire
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)  # Ferme la figure pour libérer la mémoire

    prediction_table = dataset[['flight', 'airline', 'flight_status', 'predicted_parking']].to_html(classes="table table-striped", index=False)

    return render_template("prediction.html", prediction_table=prediction_table, plot_url=img)

@app.route("/optimization", methods=["GET", "POST"])
def optimization():
    if "user" not in session:
        return redirect(url_for("index"))

    # Charger le dataset avec les prédictions
    try:
        DATA_PATH = "/Users/benothmane/Desktop/flight_dashboard/static/dataset.csv"
        dataset = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return "Erreur : Fichier non trouvé. Vérifiez le chemin du dataset."

    dataset['needs_parking'] = np.where(dataset['flight_status'] == 'landed', 1, 0)
    n_flights = len(dataset)

    # Définition du problème d'optimisation linéaire
    places_disponibles = 8
    c = np.zeros(n_flights)
    c[dataset['needs_parking'] == 1] = 1
    A_ub = np.zeros((places_disponibles, n_flights))
    b_ub = np.ones(places_disponibles)

    for i in range(places_disponibles):
        A_ub[i, :] = dataset['needs_parking']

    x_bounds = (0, 1)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x_bounds]*n_flights, method='highs')

    # Traitement des résultats d'optimisation
    results = []
    if res.success:
        allocation = res.x
        for i, alloc in enumerate(allocation):
            status = "Peut stationner" if alloc > 0.5 else "Ne peut pas stationner"
            results.append({
                "Flight ID": dataset['flight'][i],
                "Airline": dataset['airline'][i],
                "Stationnement": status
            })
    optimization_table = pd.DataFrame(results).to_html(classes="table table-striped", index=False)

    return render_template("optimization.html", optimization_table=optimization_table)

# Démarrer l'application
if __name__ == "__main__":
    app.run(debug=True)
