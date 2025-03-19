from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() #Load environment variables from .env file.

app = Flask(__name__)

# Configure your Gemini API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=GOOGLE_API_KEY)



# Select the Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')  # or 'gemini-pro-vision' for multimodal

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Generates text using the Gemini API.

    Request body (JSON):
    {
        "prompt": "tell me a joke about ai",
        "temperature": 0.9, # Optional, default 0.9
        "top_p": 1, # Optional, default 1
        "top_k": 1, # Optional, default 1
        "max_output_tokens": 2048 # optional, default 2048
    }

    Response (JSON):
    {
        "generated_text": "Generated text from Gemini"
    }
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        temperature = data.get('temperature', 0.9)
        top_p = data.get('top_p', 1)
        top_k = data.get('top_k', 1)
        max_output_tokens = data.get('max_output_tokens', 2048)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
            ),
        )

        return jsonify({"generated_text": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("PORT", 5000))) #Use environment port variable.