from flask import Blueprint, render_template, request, redirect, url_for

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@bp.route('/user/<int:user_id>', methods=['GET', 'POST'])
def regression(user_id):
    if request.method == 'POST':
        # Manejar el archivo subido o la URL del dataset
        dataset = request.files.get('dataset')
        url = request.form.get('url')
        neurons = request.form.get('neurons')
        model = request.form.get('model')
        hidden_activation = request.form.get('hidden_activation')
        output_activation = request.form.get('output_activation')
        layers = request.form.get('layers')
        epoch = request.form.get('epoch')

        # Procesar el archivo y los parámetros según tus necesidades
        # Aquí podrías agregar lógica para manejar el dataset y los parámetros

        # Redirigir después de procesar
        return redirect(url_for('main.home'))
    
    # Renderizar la plantilla en caso de GET
    return render_template('regression.html')

@bp.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        # Manejar el archivo subido o la URL del dataset
        dataset = request.files.get('dataset')
        url = request.form.get('url')
        neurons = request.form.get('neurons')
        model = request.form.get('model')
        hidden_activation = request.form.get('hidden_activation')
        output_activation = request.form.get('output_activation')
        layers = request.form.get('layers')
        epoch = request.form.get('epoch')

        # Procesar el archivo y los parámetros según tus necesidades
        # Aquí podrías agregar lógica para manejar el dataset y los parámetros

        # Redirigir después de procesar
        return redirect(url_for('main.home'))

    # Renderizar la plantilla en caso de GET
    return render_template('classification.html')

