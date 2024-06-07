from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from ..utils.data_processing import process_regression_data, train_regression_model, load_dataset

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@bp.route('/regression', methods=['GET', 'POST'])
def regression(): 
    if request.method == 'POST':
        try:
            dataset_file = request.files.get('dataset')
            url = request.form.get('url')

            if not dataset_file and not url:
                return jsonify({'status': 'error', 'message': 'No dataset file or URL provided.'})

            dataset = load_dataset(file=dataset_file, url=url)

            input_variables_text = request.form.get('input_variables_text')
            output_variable_text = request.form.get('output_variable_text')

            if not input_variables_text or not output_variable_text:
                return jsonify({'status': 'error', 'message': 'Input and output variables are required.'})

            input_cols = [col.strip() for col in input_variables_text.split(',')]
            output_col = output_variable_text.strip()

            X, y = process_regression_data(dataset, input_cols, output_col)

            model_type = request.form.get('model')
            params = {}
            if model_type == 'polynomial':
                params['degree'] = int(request.form.get('degree', 2))
            elif model_type in ['ridge', 'lasso']:
                params['alpha'] = float(request.form.get('alpha', 1.0))
            elif model_type == 'neural_network':
                params['neurons'] = int(request.form.get('neurons', 10))
                params['hidden_activation'] = request.form.get('hidden_activation', 'relu')
                params['output_activation'] = request.form.get('output_activation', 'linear')
                params['layers'] = int(request.form.get('layers', 1))
                params['epochs'] = int(request.form.get('epochs', 100))

            model, history, plot_div, redirect_url = train_regression_model(X, y, model_type, **params)

            return jsonify({'status': 'success', 'plot_div': plot_div, 'redirect': redirect_url}), 200

        except Exception as e:
            print(f"Error occurred: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return render_template('regression.html')

@bp.route('/classification', methods=['GET', 'POST'])
def classification():
    if request.method == 'POST':
        dataset = request.files.get('dataset')
        url = request.form.get('url')
        neurons = request.form.get('neurons')
        model = request.form.get('model')
        hidden_activation = request.form.get('hidden_activation')
        output_activation = request.form.get('output_activation')
        layers = request.form.get('layers')
        epoch = request.form.get('epoch')

        return redirect(url_for('main.home'))

    return render_template('classification.html')

@bp.route('/results', methods=['GET'])
def results():
    plot_div = request.args.get('plot_div')
    return render_template('results.html', plot_div=plot_div)
