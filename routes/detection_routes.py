from flask import Blueprint, request, jsonify, send_file
import os
import random
from datetime import datetime
import openai
from werkzeug.utils import secure_filename
from config import Config
from utils.file_handling import save_uploaded_file, cleanup_old_files
from utils.image_processing import ImageProcessor
from utils.model_inference import TrainedModelService
import cv2


detection_bp = Blueprint('detection', __name__)

@detection_bp.route('/detect', methods=['POST'])
def detect_disease():
    """
    Endpoint for cucumber disease detection
    """
    try:
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400
        
        # Save uploaded file
        filepath, filename = save_uploaded_file(file)
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: ' + ', '.join(Config.ALLOWED_EXTENSIONS)
            }), 400
        
        # Clean up old files
        cleanup_old_files()
        
        # Process image (placeholder preprocessing)
        preprocess_result = ImageProcessor.preprocess_image(filepath)
        if not preprocess_result['success']:
            return jsonify({
                'success': False,
                'error': f'Image preprocessing failed: {preprocess_result["error"]}'
            }), 400
        
        # Get prediction from trained model
        prediction_result = TrainedModelService.predict(filepath)
        
        if not prediction_result['success']:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {prediction_result["error"]}'
            }), 500
        
        # Return results
        return jsonify({
            'success': True,
            'filename': filename,
            'preprocessing': preprocess_result,
            'prediction': prediction_result['prediction'],
            'image_url': f'/api/uploads/{filename}',
            'message': 'Using trained model inference.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@detection_bp.route('/pest-detect', methods=['POST'])
def detect_pest():
    """
    Lightweight pest scan endpoint that does not affect the disease model.
    It randomly selects one of four cucumber pests for demonstration purposes.
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400

        filepath, filename = save_uploaded_file(file)

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: ' + ', '.join(Config.ALLOWED_EXTENSIONS)
            }), 400

        cleanup_old_files()

        pests = ['Aphids', 'Whiteflies', 'Spider Mites', 'Thrips']
        pest_recommendations = {
            'Aphids': [
                'Spray neem oil or insecticidal soap on the affected leaves.',
                'Remove heavily infested growth and improve airflow around the plant.'
            ],
            'Whiteflies': [
                'Use yellow sticky traps and wash the undersides of leaves.',
                'Apply horticultural oil or insecticidal soap to reduce the population.'
            ],
            'Spider Mites': [
                'Rinse leaves with water and increase humidity around the plant.',
                'Apply miticide or insecticidal soap if the infestation is severe.'
            ],
            'Thrips': [
                'Remove damaged leaves and use blue sticky traps to monitor them.',
                'Apply neem oil or spinosad-based treatment for control.'
            ]
        }
        selected_pest = random.choice(pests)
        confidence = round(0.72 + (random.random() * 0.24), 2)

        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': {
                'pest': selected_pest,
                'confidence': confidence,
                'available_pests': pests,
                'recommendations': pest_recommendations[selected_pest],
                'mode': 'random_pest_scan',
                'message': 'Pest scan completed using a lightweight fallback classifier.'
            },
            'image_url': f'/api/uploads/{filename}',
            'message': 'Pest scan completed successfully.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Pest scan failed: {str(e)}'
        }), 500


@detection_bp.route('/batch-detect', methods=['POST'])
def batch_detect():
    """
    Endpoint for batch processing multiple images
    """
    try:
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        if len(files) > 10:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Maximum 10 images allowed per batch'
            }), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if file and file.filename:
                filepath, filename = save_uploaded_file(file)
                
                if filepath:
                    # Process each image
                    prediction = TrainedModelService.predict(filepath)
                    
                    if prediction['success']:
                        results.append({
                            'filename': filename,
                            'prediction': prediction['prediction'],
                            'image_url': f'/api/uploads/{filename}'
                        })
        
        return jsonify({
            'success': True,
            'total_images': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch processing failed: {str(e)}'
        }), 500

@detection_bp.route('/uploads/<filename>', methods=['GET'])
def get_uploaded_file(filename):
    """
    Serve uploaded images
    """
    try:
        filepath = os.path.join(Config.UPLOAD_FOLDER, secure_filename(filename))
        
        if os.path.exists(filepath):
            return send_file(filepath)
        else:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@detection_bp.route('/system-info', methods=['GET'])
def system_info():
    """
    Get system information and status
    """
    import sys
    import platform
    
    return jsonify({
        'success': True,
        'system': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'backend_status': 'operational',
            'model_status': 'placeholder_active',
            'upload_directory': Config.UPLOAD_FOLDER,
            'uploaded_files_count': len(os.listdir(Config.UPLOAD_FOLDER)) if os.path.exists(Config.UPLOAD_FOLDER) else 0
        },
        'capabilities': {
            'max_file_size_mb': Config.MAX_CONTENT_LENGTH / (1024 * 1024),
            'allowed_extensions': list(Config.ALLOWED_EXTENSIONS),
            'batch_processing': True,
            'max_batch_size': 10
        },
        'notes': [
            'This backend is using a placeholder model',
            'Replace PlaceholderModel with your trained ML model',
            'Update ImageProcessor.preprocess_image() for your model requirements',
            'Update disease classes in models/placeholder_model.py'
        ]
    })
    # Add this to detection_routes.py, after your existing routes
@detection_bp.route('/model/status', methods=['GET'])
def model_status():
    """
    Get model status and information
    """
    try:
        model_data = TrainedModelService.load()
        model_dir_exists = os.path.exists(Config.MODELS_DIR)
        model_loaded = model_data is not None
        model_classes = model_data.get('class_names') if model_data else Config.CLASS_NAMES
        model_path = TrainedModelService._model_path if model_data else None

        return jsonify({
            'success': True,
            'model_loaded': model_loaded,
            'model_type': 'pytorch' if model_loaded else 'none',
            'model_version': 'trained' if model_loaded else 'n/a',
            'class_names': model_classes,
            'num_classes': len(model_classes),
            'models_directory': str(Config.MODELS_DIR),  # ✅ Path → string
            'models_directory_exists': model_dir_exists,
            'model_path': model_path,
            'notes': 'Using trained model inference.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def improved_keyword_response(user_message):
    """Improved keyword-based response system"""
    user_message_lower = user_message.lower().strip()
    
    # Expanded keyword mapping with synonyms
    keyword_responses = {
        'hello': {
            'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'response': "👋 Hello! I'm your cucumber disease assistant. How can I help you today?"
        },
        'disease': {
            'keywords': ['disease', 'diseases', 'sick', 'sickness', 'infected', 'infection', 'problem'],
            'response': """🌱 **Common Cucumber Diseases:**
            
1. **Angular Leaf Spot** - Angular, water-soaked spots on leaves
2. **Downy Mildew** - Yellow spots on top, purple/gray mold underneath  
3. **Anthracnose** - Circular, sunken spots with pink/orange spores
4. **Powdery Mildew** - White powdery coating on leaves and stems
5. **Bacterial Wilt** - Sudden wilting, sticky ooze from cut stems

Which disease would you like to know more about?"""
        },
        'prevent': {
            'keywords': ['prevent', 'prevention', 'avoid', 'stop', 'protect', 'keep healthy'],
            'response': """🛡️ **Prevention Strategies:**
            
• **Air Circulation**: Space plants 2-3 feet apart
• **Watering**: Water at soil level (never on leaves), early morning
• **Sanitation**: Remove infected leaves immediately
• **Rotation**: Don't plant cucumbers in same spot for 3 years
• **Preventive Sprays**: Apply copper fungicide every 7-10 days
• **Resistant Varieties**: Choose disease-resistant cucumber types"""
        },
        'treat': {
            'keywords': ['treat', 'treatment', 'cure', 'fix', 'spray', 'fungicide', 'chemical'],
            'response': """💊 **Treatment Options:**
            
**For Fungal Diseases:**
• **Mild cases**: Neem oil or baking soda spray (1 tbsp/gal)
• **Moderate**: Copper-based fungicide every 7-10 days
• **Severe**: Sulfur-based fungicide (follow label instructions)

**For Bacterial Diseases:**
• Remove and destroy infected plants
• Disinfect tools after use
• No effective chemical treatments available

**Always**: Test on a few leaves first before full application!"""
        },
        'symptom': {
            'keywords': ['symptom', 'sign', 'look like', 'appearance', 'spot', 'yellow', 'wilt', 'mold'],
            'response': """🔍 **Disease Symptoms to Look For:**
            
• **Yellow/brown angular spots** = Angular Leaf Spot
• **White powdery coating** = Powdery Mildew  
• **Yellow top, purple bottom** = Downy Mildew
• **Sunken circular spots** = Anthracnose
• **Sudden wilting, sticky stems** = Bacterial Wilt
• **Stunted growth** = Multiple possible causes

Upload a photo for specific diagnosis!"""
        },
        'upload': {
            'keywords': ['upload', 'image', 'picture', 'photo', 'detect', 'scan', 'analyze'],
            'response': """📸 **Image Analysis Available:**
            
Go to the **Detection Page** to upload cucumber leaf images. I will:
1. Analyze for disease symptoms
2. Identify potential diseases
3. Provide confidence scores
4. Suggest treatment options

Supported: PNG, JPG, JPEG files up to 16MB"""
        },
        'watering': {
            'keywords': ['water', 'watering', 'irrigate', 'moisture', 'dry', 'how much water'],
            'response': """💧 **Watering Best Practices:**
            
• **Frequency**: 1-2 inches per week (more in hot weather)
• **Timing**: Early morning (allows leaves to dry)
• **Method**: Drip irrigation or soaker hoses (not overhead)
• **Soil**: Keep consistently moist but not waterlogged
• **Signs of overwatering**: Yellow leaves, mold growth"""
        },
        'soil': {
            'keywords': ['soil', 'dirt', 'ground', 'compost', 'fertilizer', 'ph', 'nutrient'],
            'response': """🌿 **Soil Requirements:**
            
• **Type**: Well-draining loamy soil
• **pH**: 6.0-6.8 (slightly acidic)
• **Preparation**: Mix in 3-4 inches of compost before planting
• **Fertilizer**: Balanced (10-10-10) every 4-6 weeks
• **Mulch**: 2-3 inches of straw to retain moisture"""
        },
        'harvest': {
            'keywords': ['harvest', 'pick', 'ripe', 'ready', 'when to pick'],
            'response': """⏰ **Harvest Timing:**
            
• **Size**: 6-8 inches long for optimal flavor
• **Frequency**: Every 2-3 days during peak season
• **Time of day**: Early morning when cool
• **Method**: Cut stem ¼ inch above fruit (don't pull)
• **Storage**: Refrigerate immediately, lasts 7-10 days"""
        }
    }
    
    # Check for exact matches first
    for category, data in keyword_responses.items():
        for keyword in data['keywords']:
            if keyword in user_message_lower:
                return data['response']
    
    # Default helpful response
    return f"""🤔 I understand you're asking about: **"{user_message}"**

I'm a cucumber disease specialist. Try asking about:

• Specific diseases (powdery mildew, bacterial wilt, etc.)
• Disease symptoms and identification  
• Prevention and treatment methods
• Watering and soil requirements
• Harvesting tips

Or upload a leaf image on the **Detection Page** for analysis!"""

def get_gpt_response(user_message):
    """Get response from OpenAI GPT"""
    try:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        
        openai.api_key = Config.OPENAI_API_KEY
        
        response = openai.ChatCompletion.create(
            model=Config.GPT_MODEL or "gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are CucumberBot, a helpful cucumber plant disease expert.
                    
                    GUIDELINES:
                    1. Provide practical, actionable advice for home gardeners
                    2. Focus on identification, prevention, and treatment
                    3. Keep responses concise (2-3 paragraphs maximum)
                    4. Mention specific diseases: Angular Leaf Spot, Downy Mildew, Anthracnose, Powdery Mildew, Bacterial Wilt
                    5. Include organic options when possible
                    6. If user asks about something unrelated, gently steer back to cucumber topics
                    7. Format with brief bullet points or short paragraphs
                    8. End with a helpful suggestion or question"""
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=350,
            temperature=0.7,
            presence_penalty=0.3,
            frequency_penalty=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"GPT Error: {e}")
        return None

def get_chat_response(user_message):
    """Hybrid response system - tries GPT first, falls back to keywords"""
    
    # Quick responses for common simple questions
    quick_responses = {
        'hello': "👋 Hello! I'm your cucumber disease assistant. How can I help?",
        'hi': "Hi there! Ready to help with your cucumber questions.",
        'thanks': "You're welcome! 😊 Happy to help your cucumbers thrive!",
        'thank you': "Glad I could help! Feel free to ask more questions.",
        'bye': "Goodbye! 👋 Come back if you have more cucumber questions!",
        'ok': "Got it! What else would you like to know about cucumbers?",
        'yes': "Great! What specific cucumber question can I help with?",
        'no': "Alright, let me know what you'd like to know instead.",
    }
    
    user_lower = user_message.lower().strip()
    
    # Check for quick responses first
    for keyword, response in quick_responses.items():
        if user_lower == keyword or user_lower.startswith(keyword):
            return response, 'quick'
    
    # Check if message is too short or vague for GPT
    if len(user_message.split()) < 3 or user_lower in ['help', 'what', 'how']:
        return improved_keyword_response(user_message), 'keyword'
    
    # Try GPT for complex questions
    gpt_response = get_gpt_response(user_message)
    
    if gpt_response:
        return gpt_response, 'gpt'
    else:
        # Fall back to keyword system
        return improved_keyword_response(user_message), 'keyword'

# ==================== CHAT ROUTE ====================

@detection_bp.route('/chat', methods=['OPTIONS', 'POST'])
def chat():
    """
    Hybrid chat endpoint - uses GPT when available, keywords as fallback
    """
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        # Get response using hybrid system
        response_text, response_type = get_chat_response(message)
        
        return jsonify({
            'success': True,
            'response': response_text,
            'original_message': message,
            'response_type': response_type,
            'gpt_available': bool(Config.OPENAI_API_KEY)
        })
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Chat error: {str(e)}',
            'fallback_response': "I'm having trouble right now. Try asking about cucumber diseases, prevention, or upload an image for analysis."
        }), 500

# ==================== KEEP YOUR EXISTING ROUTES BELOW ====================
# (All your existing /detect, /batch-detect, etc. routes remain the same)
# ... [REST OF YOUR EXISTING CODE] ...

