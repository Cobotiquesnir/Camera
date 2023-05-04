import mysql.connector

# Connexion à la base de données
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="xyz"
)

# Création d'un curseur
cur = conn.cursor()

# Suppression des données de la table
cur.execute("DELETE FROM co")

# Validation des changements
conn.commit()

# Fermeture de la connexion
cur.close()
conn.close()
