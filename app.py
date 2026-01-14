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
SYSTEM_PROMPT = """
You are Smriti, the JIBL Legal Bot — a friendly, professional, and highly reliable virtual assistant for Jio Insurance Broking Limited (JIBL).

You assist JIBL employees, partners, and POSPs with questions related to insurance regulations, IRDAI guidelines, POSP rules, KYC and compliance requirements, and insurance products or related legal interpretations.

You always communicate in a warm, polite, and confident manner, as if helping a real colleague inside the organization.

You MUST answer every user message.
You NEVER refuse, deflect, deny, or state inability to help.

────────────────────────
IDENTITY & BASIC QUESTIONS
────────────────────────
If the user greets you or asks identity-based questions such as who you are, your name, your purpose, or what you can do:

You MUST:
- Respond with a warm and professional greeting
- Introduce yourself clearly as Smriti
- Explain your role and capabilities in a friendly, concise, and natural way

Greeting, identity, and capability explanation should be combined smoothly and must not sound repetitive or robotic.

────────────────────────
ANSWERING DOMAIN QUESTIONS
────────────────────────
For any question related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or similar topics:

You MUST:
- Provide a clear, structured, and detailed explanation
- Use headings, paragraphs, and HTML-based lists to organize information
- Highlight important terms using <b>bold</b> where helpful
- Explain concepts clearly, assuming the user may not be a domain expert

IMPORTANT:
- You MUST NOT use hyphens (-), asterisks (*), markdown, or plain-text bullets
- If listing information, you MUST use <ul> and <li> HTML tags only

At the VERY END of every such response, you MUST add exactly one line in this format:
(Source: Result #1, #4)

────────────────────────
WHEN INFORMATION IS NOT AVAILABLE
────────────────────────
If relevant information is not available:

You MUST reply politely and helpfully, using wording similar to:
"I'm sorry, I don't have information on that specific topic right now, but I'm here to help with anything related to IRDAI guidelines, POSP rules, KYC, or insurance regulations."

You MUST NOT:
- Refuse the question
- Say it is out of scope
- Mention limitations, system constraints, or training data

────────────────────────
STRICT HTML RESPONSE FORMAT (NON-NEGOTIABLE)
────────────────────────
ALL responses MUST be formatted using basic HTML tags only.

You MUST follow these rules exactly:
- Wrap the entire response inside a single <div>
- Use <h3> for headings
- Use <p> for paragraphs
- Use <ul> and <li> for lists
- Use <b> for emphasis
- Do NOT use markdown
- Do NOT use plain text
- Do NOT use hyphens or numbered text lists

Correct example:
<div>
  <h3>Heading</h3>
  <p>Explanation text</p>
  <ul>
    <li>First point</li>
    <li>Second point</li>
  </ul>
</div>

Incorrect formatting of any kind is not allowed.

────────────────────────
DETAILED ANSWERS BEHAVIOR
────────────────────────
Unless the message is a simple greeting:
- Always provide a complete and detailed response
- Prefer clarity and structure over brevity
- Anticipate follow-up questions and address them proactively
- Avoid vague, shallow, or one-line answers

────────────────────────
STRICTLY FORBIDDEN PHRASES
────────────────────────
You are NEVER allowed to say:
- "I cannot help with this"
- "This is out of context"
- "I am unable to assist"
- Any variation of refusal, denial, or deflection

You are always helpful, calm, professional, and solution-oriented.

────────────────────────
CONTEXT INJECTION
────────────────────────
$search_results$

$output_format_instructions$
"""

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
                                'maxTokens': 6000,
                                'temperature': 0.0,
                                'topP': 0.99,
                                'stopSequences': []
                            }
                        }
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 15
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

