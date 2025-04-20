from flask import Flask, render_template
import os

# chemin absolu vers le dossier templates
template_dir = os.path.abspath("C:/Users/Utilisateur/PycharmProjects/Flask_chat_chien/templates")
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
