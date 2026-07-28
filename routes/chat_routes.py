from flask import Blueprint, request, jsonify
import openai
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

chat_bp = Blueprint('chat', __name__)

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# System prompt for plant disease expert
PLANT_EXPERT_SYSTEM = """You are "Dr. Green", a senior plant pathologist with 25 years of experience. 
You specialize in diagnosing plant diseases and providing practical treatment advice.

YOUR ROLE:
1. **Diagnosis Expert** - Identify diseases from symptoms
2. **Treatment Advisor** - Recommend organic and chemical solutions
3. **Prevention Specialist** - Suggest preventive measures
4. **Emergency Responder** - Flag urgent cases needing immediate action

RESPONSE FORMAT:
🌿 **Diagnosis**: Clear identification
💊 **Treatment**: Step-by-step instructions
🛡️ **Prevention**: Long-term strategies
⚠️ **Urgency Level**: Low/Medium/High
📋 **Follow-up**: What to monitor

TONE: Professional but friendly, empathetic, practical
LANGUAGE: Simple English, avoid jargon unless explaining
LENGTH: 200-400 words, use bullet points for clarity"""

@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        # Prepare conversation for ChatGPT
        messages = [
            {"role": "system", "content": PLANT_EXPERT_SYSTEM}
        ]
        
        # Add conversation history (last 5 messages to stay within token limits)
        for msg in conversation_history[-5:]:
            messages.append({
                "role": "user" if msg.get('sender') == 'user' else "assistant",
                "content": msg.get('text', '')
            })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Call ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" for better results
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract the response
        bot_response = response.choices[0].message.content
        
        # Log the interaction (optional)
        log_interaction(user_message, bot_response)
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'model': response.model,
            'tokens_used': response.usage.total_tokens
        })
        
    except openai.error.AuthenticationError:
        return jsonify({
            'success': False,
            'error': 'Invalid API key. Please check your OpenAI credentials.'
        }), 401
        
    except openai.error.RateLimitError:
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please try again in a moment.'
        }), 429
        
    except openai.error.OpenAIError as e:
        return jsonify({
            'success': False,
            'error': f'OpenAI API error: {str(e)}'
        }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

def log_interaction(user_msg, bot_response):
    """Log chat interactions for debugging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}]\nUser: {user_msg}\nBot: {bot_response[:100]}...\n"
    
    with open('chat_logs.txt', 'a', encoding='utf-8') as f:
        f.write(log_entry)

# Additional route for getting conversation history
@chat_bp.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for a user (simplified - in production, use database)"""
    user_id = request.args.get('user_id')
    
    # In production, fetch from database
    # For now, return empty or mock data
    return jsonify({
        'success': True,
        'history': []
    })

# Route to test API connection
@chat_bp.route('/api/chat/test', methods=['GET'])
def test_connection():
    """Test if ChatGPT API is working"""
    try:
        # Make a simple test call
        test_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Plant Health API is working!'"}],
            max_tokens=10
        )
        
        return jsonify({
            'success': True,
            'message': test_response.choices[0].message.content,
            'status': 'API is working correctly'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500