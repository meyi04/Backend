from flask import Blueprint, request, jsonify, send_file
import os
import re
import logging
from datetime import datetime
import openai
from werkzeug.utils import secure_filename
from config import Config
from utils.file_handling import save_uploaded_file, cleanup_old_files
from utils.image_processing import ImageProcessor
from utils.model_inference import TrainedModelService
from utils.roboflow_pest_inference import RoboflowPestService
import cv2


detection_bp = Blueprint('detection', __name__)
logger = logging.getLogger(__name__)

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
    Run the Roboflow cucumber pest detection model.
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

        prediction = RoboflowPestService.predict(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': prediction,
            'image_url': f'/api/uploads/{filename}',
            'message': 'Pest scan completed using Roboflow.'
        })

    except Exception as e:
        logger.exception("Roboflow pest inference failed")
        return jsonify({
            'success': False,
            'error': f'Pest scan failed: {str(e)}'
        }), 502


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
        'aphids': {
            'keywords': ['aphid', 'aphids'],
            'response': """🐜 **Aphid Treatment:**

1. Spray the undersides of leaves and new growth with a strong stream of water.
2. Prune heavily infested tips and dispose of them away from the compost.
3. Use yellow sticky traps and protect ladybugs and other beneficial insects.
4. If the infestation continues, use insecticidal soap or another product labelled for aphids and cucumbers. Follow the label and test a small area first.
5. Recheck the plant every few days because aphids reproduce quickly.

Sticky honeydew, curled leaves, and ants may also indicate aphids. """
        },
        'pests': {
            'keywords': ['pest', 'pests', 'insect', 'insects', 'bug', 'bugs'],
            'response': """🐛 **Cucumber Pest Treatment:**

• Inspect the undersides of leaves for aphids, whiteflies, thrips, and spider mites.
• Look for feeding holes and striped or spotted cucumber beetles.
• Remove visible pests by hand or rinse leaves with water.
• Use sticky traps, row covers, and weed control to reduce pest pressure.
• Protect beneficial insects and use a labelled treatment only when monitoring confirms an infestation.

Tell me what the insect looks like or share a clear photo so I can narrow it down."""
        },
        'whiteflies': {
            'keywords': ['whitefly', 'whiteflies', 'white fly', 'white flies'],
            'response': """🪰 **Whitefly Treatment:**

• Check the undersides of leaves for tiny white insects, pale nymphs, and sticky honeydew.
• Use yellow sticky traps to monitor adults and rinse leaf undersides with water.
• Remove weeds and badly infested leaves, but keep ladybugs and other predators.
• If pressure remains high, use a labelled insecticidal soap or horticultural oil and repeat only as directed on the label.
• Recheck new growth every few days."""
        },
        'thrips': {
            'keywords': ['thrips', 'silver streaks', 'silvery streaks'],
            'response': """🦟 **Thrips Treatment:**

• Look for silvery scars, black specks, and distorted young leaves or flowers.
• Remove weeds and damaged plant material and use blue or yellow sticky traps.
• Rinse foliage gently and protect predatory mites and other beneficial insects.
• Use only a product labelled for thrips and cucumbers when monitoring confirms an infestation; rotate approved modes of action according to the label."""
        },
        'spider_mites': {
            'keywords': ['spider mite', 'spider mites', 'mite', 'mites', 'webbing'],
            'response': """🕷️ **Spider Mite Treatment:**

• Check for fine webbing and tiny pale speckles, especially on lower leaf surfaces.
• Isolate the plant and rinse the undersides of leaves with water.
• Reduce dusty, very dry conditions without keeping foliage wet overnight.
• Encourage predatory mites and avoid broad-spectrum insecticides, which can worsen outbreaks.
• Use a labelled miticide or horticultural oil only if the infestation persists and the label permits use on cucumbers."""
        },
        'cucumber_beetles': {
            'keywords': ['cucumber beetle', 'cucumber beetles', 'striped beetle', 'spotted beetle'],
            'response': """🪲 **Cucumber Beetle Treatment:**

• Hand-pick adults or vacuum them carefully from the plant.
• Protect young plants with row covers, removing covers during flowering for pollination.
• Remove weeds and plant debris where beetles shelter.
• Monitor for feeding holes and wilting because beetles can spread bacterial wilt.
• Use a labelled treatment only when damage is increasing and follow pollinator-safety directions."""
        },
        'cutworms': {
            'keywords': ['cutworm', 'cutworms', 'seedling cut', 'cut at the soil'],
            'response': """🐛 **Cutworm Treatment:**

• Inspect around the soil line after dark and hand-remove caterpillars.
• Clear weeds and debris before planting.
• Place a paper or cardboard collar around each seedling, pushed slightly into the soil.
• Keep the growing area clean and check young plants daily.
• Use a labelled product only for confirmed damage and follow harvest and safety intervals."""
        },
        'downy_mildew': {
            'keywords': ['downy mildew', 'purple mold', 'gray mold underneath', 'grey mold underneath'],
            'response': """🍃 **Downy Mildew Treatment:**

• Confirm yellow angular spots on top of leaves and gray-purple growth underneath.
• Remove badly affected leaves and bag them; do not compost suspected disease.
• Improve spacing and airflow and water at soil level.
• Avoid working among wet plants to reduce spread.
• Use a fungicide labelled for downy mildew on cucumbers as soon as possible; follow the label and rotate product groups as directed."""
        },
        'powdery_mildew': {
            'keywords': ['powdery mildew', 'white powder', 'white coating', 'flour on leaves'],
            'response': """🌿 **Powdery Mildew Treatment:**

• Remove the most heavily affected leaves and disinfect tools.
• Improve airflow, reduce crowding, and avoid excessive nitrogen fertilizer.
• Keep plants adequately watered at the soil, but avoid wetting foliage overnight.
• Use a labelled sulfur, biological, or other approved fungicide early; test products on a small area first and follow the label."""
        },
        'angular_leaf_spot': {
            'keywords': ['angular leaf spot', 'water soaked spots', 'water-soaked spots', 'yellow halos'],
            'response': """🟤 **Angular Leaf Spot Treatment:**

• Remove leaves with angular, water-soaked lesions and yellow halos.
• Avoid overhead irrigation and do not handle plants while wet.
• Disinfect tools and remove infected debris; do not save seed from affected plants.
• Improve airflow and use resistant varieties when available.
• A labelled copper product may help suppress bacterial spread, but it will not restore damaged tissue."""
        },
        'anthracnose': {
            'keywords': ['anthracnose', 'sunken lesions', 'pink spores', 'fruit rot'],
            'response': """🟠 **Anthracnose Treatment:**

• Remove infected leaves and fruit and dispose of them away from the garden.
• Keep foliage and fruit dry, improve airflow, and avoid overhead irrigation.
• Rotate cucurbit crops and clean plant debris after harvest.
• Use a labelled fungicide early when anthracnose is confirmed and follow the harvest interval on the label."""
        },
        'bacterial_wilt': {
            'keywords': ['bacterial wilt', 'sudden wilt', 'sticky ooze', 'wilting vine'],
            'response': """⚠️ **Bacterial Wilt Treatment:**

• There is no reliable cure for an infected cucumber plant.
• Remove and bag the whole plant promptly; do not compost it.
• Control cucumber beetles, which commonly spread the bacteria.
• Disinfect tools and wash hands before touching healthy plants.
• Use resistant varieties and row covers next season."""
        },
        'mosaic_virus': {
            'keywords': ['mosaic virus', 'mosaic', 'mottled leaves', 'virus'],
            'response': """🧬 **Mosaic Virus Management:**

• Mottled leaves, distortion, and stunting can indicate a virus, but symptoms can overlap with other stress.
• Remove and bag confirmed infected plants because there is no curative fungicide.
• Control aphids and remove weeds that can host the virus.
• Disinfect tools and wash hands; avoid spreading sap between plants.
• Use certified seed and resistant varieties where available."""
        },
        'yellow_leaves': {
            'keywords': ['yellow leaves', 'leaves turning yellow', 'yellowing leaves'],
            'response': """💛 **Yellow Leaves:**

Yellowing can result from overwatering, nutrient stress, aging leaves, spider mites, downy mildew, or root problems.

• Check soil moisture and drainage before adding fertilizer.
• Inspect both sides of leaves for pests and lesions.
• Look for a pattern: older leaves, new leaves, veins, or scattered spots.
• Improve airflow and water at soil level.

Tell me whether the yellowing starts on old or new leaves and whether spots, insects, or webbing are present."""
        },
        'leaf_holes': {
            'keywords': ['holes in leaves', 'holes on leaves', 'chewed leaves', 'eaten leaves', 'feeding holes'],
            'response': """🕳️ **Holes or Chewed Leaves:**

• Inspect the plant at dusk and check under leaves for beetles, caterpillars, slugs, or eggs.
• Hand-pick visible pests and remove nearby weeds and debris.
• Use row covers on young plants and traps where appropriate.
• Do not spray based on holes alone; identify the pest first so beneficial insects are not harmed.
• Share a photo of the damage and the insect if you see one."""
        },
        'leaf_curling': {
            'keywords': ['curling leaves', 'curled leaves', 'leaves curling', 'twisted leaves'],
            'response': """🔄 **Curling Leaves:**

Possible causes include aphids, whiteflies, thrips, heat or drought stress, herbicide drift, and viral disease.

• Inspect new growth and leaf undersides for insects and honeydew.
• Check soil moisture and protect the plant from extreme heat.
• Remove badly damaged leaves and avoid applying unlabelled sprays.
• If curling is paired with mosaic patterns or severe stunting, remove the plant and prevent sap spread."""
        },
        'root_rot': {
            'keywords': ['root rot', 'rotting roots', 'roots are brown', 'waterlogged'],
            'response': """🌱 **Possible Root Rot:**

• Check whether soil stays wet and whether roots are brown, soft, or foul-smelling.
• Improve drainage, reduce watering, and do not let containers sit in water.
• Remove severely affected plants and sanitize containers and tools.
• Use clean, well-draining growing media and avoid planting too deeply.
• Yellowing and wilting can have several causes, so confirm the roots before treating."""
        },
        'organic_treatment': {
            'keywords': ['organic treatment', 'organic option', 'natural treatment', 'home remedy', 'neem oil'],
            'response': """🌿 **Lower-Risk Treatment Options:**

• Start with hand removal, water rinsing, pruning, traps, row covers, sanitation, and beneficial insects.
• Insecticidal soap or horticultural oil can help some soft-bodied pests when the label permits.
• Biological fungicides may help suppress some diseases but do not cure advanced infections.
• Test any product on a small area first, spray according to the label, and avoid spraying flowers or active pollinators.
• Natural does not automatically mean safe; never mix household products or use an unlabelled recipe on edible crops."""
        },
        'spray_safety': {
            'keywords': ['safe pesticide', 'safe spray', 'how much pesticide', 'mix pesticide', 'spray leaves'],
            'response': """🧴 **Spray Safety:**

• Identify the pest or disease before spraying.
• Use only a product labelled for cucumbers and the target problem.
• Follow the label for dose, protective equipment, weather, re-entry, and harvest interval.
• Never mix products unless the label specifically permits it.
• Avoid windy or very hot conditions and protect bees and beneficial insects.
• Do not use household cleaners, fuels, or unlabelled chemicals on edible plants."""
        },
        'disease_spread': {
            'keywords': ['spreading', 'spread quickly', 'stop spreading', 'contagious', 'infect other plants'],
            'response': """🚧 **Stopping Spread:**

• Isolate the affected plant when possible.
• Remove infected leaves or the whole plant when bacterial or viral disease is suspected.
• Bag diseased material and do not compost it.
• Water at soil level, improve airflow, and avoid working among wet plants.
• Disinfect pruners between plants and wash hands.
• Control insect vectors such as aphids and cucumber beetles."""
        },
        'diagnosis_help': {
            'keywords': ['what disease', 'identify disease', 'diagnose', 'is this a disease', 'what is wrong'],
            'response': """🔎 **For Better Identification:**

Please provide the plant age, affected part, first symptom, speed of spread, recent weather, watering pattern, and a clear photo of the whole plant plus close-ups of both leaf surfaces.

I can compare likely disease, pest, nutrient, and watering causes, but symptoms alone cannot confirm a diagnosis. Do not spray until the likely cause is narrowed down."""
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
    
    # Check specific topics before greetings, and match words rather than
    # arbitrary substrings (for example, "hi" is inside "aphids").
    category_order = [
        'aphids', 'whiteflies', 'thrips', 'spider_mites', 'cucumber_beetles',
        'cutworms', 'downy_mildew', 'powdery_mildew', 'angular_leaf_spot',
        'anthracnose', 'bacterial_wilt', 'mosaic_virus', 'yellow_leaves',
        'leaf_holes', 'leaf_curling', 'root_rot', 'organic_treatment',
        'spray_safety', 'disease_spread', 'diagnosis_help', 'pests', 'treat',
        'symptom', 'disease', 'prevent', 'upload', 'watering', 'soil',
        'harvest', 'hello'
    ]
    for category in category_order:
        data = keyword_responses[category]
        for keyword in data['keywords']:
            pattern = rf'(?<!\w){re.escape(keyword)}(?!\w)'
            if re.search(pattern, user_message_lower):
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
