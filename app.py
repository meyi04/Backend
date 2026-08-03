from flask import Flask, jsonify
from flask_cors import CORS
import logging
from config import Config
from routes.detection_routes import detection_bp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # FIXED: Complete CORS configuration with proper closing
    CORS(app, origins=["http://localhost:3000", "http://localhost:5173",
    "http://127.0.0.1:5173"])
    
    # Load configuration
    app.config.from_object(Config)
    
    # Initialize directories
    Config.init_app()
    
    # Register blueprints
    app.register_blueprint(detection_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'Cucumber Disease Detection API',
            'version': '1.0.0'
        })
    
    @app.route('/')
    def index():
        return jsonify({
            'message': 'Cucumber Disease Detection API',
            'endpoints': {
                'health': '/health',
                'detect': '/api/detect',
                'model_status': '/api/model/status',
                'upload_dataset': '/api/upload/dataset'
            }
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    logger.info("🚀 Starting Cucumber Disease Detection Backend...")
    logger.info(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    logger.info(f"🤖 Models directory: {Config.MODELS_DIR}")
    logger.info("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

   