from flask import Flask, render_template, request, jsonify, make_response
import boto3
import os

# No load_dotenv() needed—App Runner injects env vars directly
app = Flask(__name__)

# === Secure: Load from App Runner environment variables ===
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-south-1")
KB_ID                 = os.environ.get("KB_ID")
MODEL_ARN             = os.environ.get("MODEL_ARN")

# Fail fast if missing (logs error in App Runner console)
required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "KB_ID", "MODEL_ARN"]
missing = [var for var in required_vars if not os.environ.get(var)]
if missing:
    raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

# Bedrock client setup
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
client = session.client("bedrock-agent-runtime", region_name=AWS_REGION)
SYSTEM_PROMPT = """You are Smriti, the JIBL Legal Bot - a friendly, professional and extremely helpful assistant for Jio Insurance Broking Limited.

You are talking to real employees, partners and POSPs, so you are always warm, polite and ready to help with anything they ask.

You MUST answer every single message. Never refuse, never say you cannot help.

Specific instructions you always follow:

1. If the user says Hi, Hello, Hii, Hey, Good morning, Good evening, What is your name, Who are you, etc.:
   Reply with a warm greeting and introduce yourself.
   Example responses (use any of these or similar):
   "Hello! I'm Smriti, the JIBL Legal Bot. How can I help you today?"
   "Hi there! This is Smriti - your insurance regulation assistant. What would you like to know?"
   "Hey! I'm Smriti, happy to help with any IRDAI, POSP or compliance questions!"

2. For any question related to insurance, IRDAI guidelines, POSP rules, KYC, products, or regulations:
   Answer clearly and professionally using bullet points, tables or bold text when it helps.
   At the very end of the answer add one line exactly like this:
   (Source: Result #1, #4)

3. If there is no relevant information available for the question:
   Reply politely:
   "I'm sorry, I don't have information on that specific topic right now, but I'm here to help with anything related to IRDAI guidelines, POSP rules, KYC or insurance regulations!"

You are NEVER allowed to say:
- "Sorry, I am unable to assist you with this request"
- "This query is out of context"
- or any kind of refusal

You are always friendly and conversational.

$search_results$

$output_format_instructions$"""

@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    # Add cache-busting headers to prevent browser caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        response = client.retrieve_and_generate(
            input={'text': user_message},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KB_ID,
                    'modelArn': MODEL_ARN,
                    'generationConfiguration': {
                        'promptTemplate': {
                            'textPromptTemplate': SYSTEM_PROMPT
                        },
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': 4096,
                                'temperature': 0.0,
                                'topP': 0.99,
                                'stopSequences': []
                            }
                        }
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 10
                        }
                    }
                }
            }
        )

        bot_reply = response['output']['text']

        # === FIX: Remove fake sources on greetings & casual questions ===
        greeting_keywords = [
            "hi", "hello", "hii", "hey", "good morning", "good afternoon", "good evening",
            "what is your name", "who are you", "what can you do", "your name", "how are you"
        ]
        user_lower = user_message.lower().strip()

        if any(keyword in user_lower for keyword in greeting_keywords):
            sources = []                                            # No sources for greetings
        else:
            # Only extract real sources when it's a proper question
            sources = []
            for i, citation in enumerate(response.get('citations', []), 1):
                for ref in citation.get('retrievedReferences', []):
                    content = ref['content'].get('text', '')[:500] + ("..." if len(ref['content'].get('text', '')) > 500 else "")
                    uri = ref.get('location', {}).get('s3Location', {}).get('uri', 'N/A')
                    sources.append({
                        "id": i,
                        "snippet": content,
                        "file": uri
                    })

        return jsonify({
            "reply": bot_reply,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use port 5001 to avoid AirPlay conflict on macOS
    app.run(host="0.0.0.0", port=5001, debug=True)
