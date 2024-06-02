from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuraciones de la aplicación
    app.config.from_object('config.Config')
    
    with app.app_context():
        # Importar Blueprints
        from .controllers import user_controller
        # Registrar Blueprints
        app.register_blueprint(user_controller.bp)
        
    return app
