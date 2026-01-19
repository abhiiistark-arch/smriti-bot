from flask import Flask, render_template, request, jsonify, make_response
import boto3
import json
import os
from urllib.parse import urlparse, unquote

app = Flask(__name__)

# ── Configuration ───────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-south-1")
KB_ID                 = os.environ.get("KB_ID")
MODEL_ARN             = os.environ.get("MODEL_ARN")
GUARDRAIL_ID          = os.environ.get("GUARDRAIL_ID")
GUARDRAIL_VERSION     = os.environ.get("GUARDRAIL_VERSION")

required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "KB_ID", "MODEL_ARN", "GUARDRAIL_ID", "GUARDRAIL_VERSION"]
missing = [var for var in required_vars if not os.environ.get(var)]
if missing:
    raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

# Create boto3 session and clients
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Client for Knowledge Base retrieval
agent_runtime = session.client("bedrock-agent-runtime", region_name=AWS_REGION)

# Client for direct model inference
bedrock_runtime = session.client("bedrock-runtime", region_name=AWS_REGION)

# Client for S3 operations
s3_client = session.client("s3", region_name=AWS_REGION)

# Your system prompt (unchanged)
SYSTEM_PROMPT = """
You are Smriti, the JIBL Legal Bot — a friendly, professional, and highly reliable virtual assistant for Jio Insurance Broking Limited (JIBL).
You are a smart and intelligent personal assistant who can assist JIBL employees, partners, and POSPs with questions related to insurance regulations, IRDAI guidelines, POSP rules, KYC and compliance requirements, and insurance products or related legal interpretations.
You always communicate in a warm, polite, and confident manner, as if helping a real colleague inside the organization.
Always provide detailed answers along with the meaningful explanation.
You MUST answer  every user message. If user asks a question outside knowledge base , you must say that you can't answer this question and you are here to help with the questions related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or similar topics.
If user asks about POSH, or any other quesions related to sexual activity, rape or harassment, just say that you are not allowed to answer this question and you are here to help with the questions related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or similar topics.
You MUST provide a clear, structured, and detailed explanation for any question related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or similar topics.
Make sure that all the subheadings are in bold and a proper indentation is maintained.
Always make sure to maintain a format for answers and must make sure all the bold parts or subheadings should start from new line.
You NEVER refuse, deflect, deny, or state inability to help.
Do not frequently use the word certainly, start the answer with different formally presentable words.
IDENTITY & BASIC QUESTIONS:-
If the user greets you or asks identity-based questions such as who you are, your name, your purpose, or what you can do:
You MUST:
- Respond with a warm and professional greeting
- Introduce yourself clearly as Smriti
- Explain your role and capabilities in a friendly, concise, and natural way
Greeting, identity, and capability explanation should be combined smoothly and must not sound repetitive or robotic.

ANSWERING DOMAIN QUESTIONS:-
For any question related to insurance, IRDAI regulations, POSP rules, KYC, compliance, or similar topics Provide a clear, structured, and detailed explanation
WHEN INFORMATION IS NOT AVAILABLE:-
If relevant information is not available:
You MUST reply politely and helpfully, using wording similar to:
"I'm sorry, I don't have information on that specific topic right now, but I'm here to help with anything related to IRDAI guidelines, POSP rules, KYC, or insurance regulations."

You MUST NOT:
- Refuse the question
- Say it is out of scope
- Mention limitations, system constraints, or training data

Unless the message is a simple greeting:
- Always provide a complete and detailed response
- Prefer clarity and structure over brevity
- Anticipate follow-up questions and address them proactively
- Avoid vague, shallow, or one-line answers

STRICTLY FORBIDDEN PHRASES
You are NEVER allowed to say:
- "I cannot help with this"
- "This is out of context"
- "I am unable to assist"
- Any variation of refusal, denial, or deflection

You are always helpful, calm, professional, and solution-oriented.

$search_results$
$output_format_instructions$

-Always break down the answers if it contains multiple parts.
-Highlight important terms using bold where helpful
-If listing information, you MUST use points to answers that.
"""

@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])   # comes as [{"role": "...", "content": "..."}, ...]

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        # 1. Retrieve from Knowledge Base (unchanged)
        retrieve_response = agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': user_message},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 7
                }
            }
        )

        retrieved_chunks = retrieve_response.get('retrievalResults', [])

        search_results_block = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk['content']['text']
            uri = chunk.get('location', {}).get('s3Location', {}).get('uri', 'N/A')
            search_results_block += f"[Result #{i}]\nContent:\n{text}\nSource: {uri}\n\n"

        # 2. Prepare messages for Bedrock Converse
        messages = []

        # Convert frontend history format → Bedrock format
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if not role or not content:
                continue
            # Normalize role names
            bedrock_role = "user" if role.lower() in ["user", "human"] else "assistant"
            messages.append({
                "role": bedrock_role,
                "content": [{"text": content}]
            })

        # Add current user message (last one)
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })

        # 3. Call model
        system_prompt_final = SYSTEM_PROMPT.replace("$search_results$", search_results_block)

        response = bedrock_runtime.converse(
            modelId=MODEL_ARN,
            messages=messages,                 # ← now correctly formatted
            system=[{"text": system_prompt_final}],
            inferenceConfig={
                "maxTokens": 2048,
                "temperature": 0.5,
                "topP": 1.0
            },
            # guardrailConfig={
            #     "guardrailIdentifier": GUARDRAIL_ID,
            #     "guardrailVersion": GUARDRAIL_VERSION,
            # }
        )

        bot_reply = response["output"]["message"]["content"][0]["text"]

        # Sources logic (unchanged)
        greeting_keywords = ["hi", "hello", "hii", "hey", "good morning", "good afternoon",
                             "good evening", "what is your name", "who are you", "what can you do",
                            "your name", "how are you"]   # your list

        show_sources = not any(kw in user_message.lower().strip() for kw in greeting_keywords)

        sources = []
        if show_sources:
            for i, chunk in enumerate(retrieved_chunks, 1):
                text = chunk['content']['text']
                snippet = text[:500] + ("..." if len(text) > 500 else "")
                uri = chunk.get('location', {}).get('s3Location', {}).get('uri', 'N/A')
                sources.append({"id": i, "snippet": snippet, "file": uri})

        return jsonify({
            "reply": bot_reply,
            "sources": sources
        })

    except Exception as e:
        import traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/presigned-url", methods=["POST"])
def get_presigned_url():
    """Generate a presigned URL for an S3 object (5 minutes expiry)"""
    try:
        data = request.get_json()
        s3_url = data.get("s3_uri", "").strip()
        
        if not s3_url:
            return jsonify({"error": "Invalid S3 URI/URL"}), 400
        
        bucket_name = None
        object_key = None
        
        # Handle s3:// URI format: s3://bucket-name/key/path
        if s3_url.startswith("s3://"):
            s3_uri = s3_url[5:]  # Remove "s3://"
            parts = s3_uri.split("/", 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else ""
        
        # Handle HTTPS S3 URL format: https://bucket.s3.region.amazonaws.com/key/path
        elif s3_url.startswith("https://") or s3_url.startswith("http://"):
            parsed = urlparse(s3_url)
            # Extract bucket from netloc (e.g., "jibllegalbottest.s3.ap-south-1.amazonaws.com" -> "jibllegalbottest")
            if ".s3." in parsed.netloc:
                bucket_name = parsed.netloc.split(".s3.")[0]
            else:
                return jsonify({"error": "Invalid S3 HTTPS URL format"}), 400
            # Decode key (%20 → space, etc.) and remove leading slash
            object_key = unquote(parsed.path.lstrip("/"))
        
        else:
            return jsonify({"error": "Invalid S3 URI/URL format. Must start with 's3://' or 'https://'"}), 400
        
        if not bucket_name or not object_key:
            return jsonify({"error": "Invalid bucket or object key"}), 400
        
        # Generate presigned URL with 5 minutes (300 seconds) expiry
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket_name,
                "Key": object_key,
                "ResponseContentDisposition": "inline"
            },
            ExpiresIn=300  # 5 minutes
        )
        
        return jsonify({"presigned_url": presigned_url})
    
    except Exception as e:
        import traceback
        print("Error generating presigned URL:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
