from flask import Flask, request, render_template, url_for
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paramètres pour les fichiers téléchargés
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Créer le dossier uploads s'il n'existe pas
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def modele(X_train, W, b):
    Z = np.dot(X_train, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A, Z

# Fonction pour charger ton modèle
def load_model():
    W = np.load('W.npy')  # Ton modèle de poids
    b = np.load('b.npy')  # Ton biais
    return W, b

# Fonction de prédiction
from PIL import Image


from PIL import Image
import numpy as np

def predict_image(image_path, W, b):
    # Étape 1-2-3 : Ouvrir, redimensionner, passer en niveaux de gris
    image = Image.open(image_path).convert('L').resize((64, 64))

    # Étape 4 : Normalisation [0,1]
    image = np.array(image) / 255.0

    # Étape 5 : Binarisation
    image = (image >= 0.5).astype(np.float32)

    # Étape 6 : Flatten
    image = image.reshape(1, -1)

    # Prédiction
    A, _ = modele(image, W, b)
    print(f"Valeur de A : {A}")

    return "Chat" if A >= 0.5 else "Chien"



# Page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Aucun fichier sélectionné'
        file = request.files['file']
        if file.filename == '':
            return 'Aucun fichier sélectionné'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            W, b = load_model()
            prediction = predict_image(file_path, W, b)
            image_url = os.path.join('uploads', filename)
            return render_template('index.html', prediction=prediction, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
