import pandas as pd

# Charger le fichier CSV
file_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Predictions_data.csv'
results_df = pd.read_csv(file_path)

# Afficher les 10 premières lignes pour vérifier les données
print("Affichage des 10 premières lignes :")
print(results_df.head(10))

# Vérifier les noms des colonnes
print("Noms des colonnes :")
print(results_df.columns)

# Supprimer les colonnes dupliquées si nécessaire
results_df = results_df.loc[:, ~results_df.columns.duplicated()]

# Sauvegarder le fichier nettoyé dans un nouveau fichier CSV
cleaned_file_path = '/Users/benothmane/Desktop/AIR_OPTIMISATION/upload/folder/Cleaned_Predictions_data.csv'
results_df.to_csv(cleaned_file_path, index=False)

# Afficher le message de confirmation
print(f"Fichier nettoyé sauvegardé sous : {cleaned_file_path}")
