from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuraciones de la aplicación
    app.config.from_object('config.Config')
    
    with app.app_context():
        # Importar Blueprints
        from .routes import user_routes
        # Registrar Blueprints
        app.register_blueprint(user_routes.bp)
        
    return app
