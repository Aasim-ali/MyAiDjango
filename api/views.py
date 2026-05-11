from google import genai
from google.genai import types
from rest_framework.decorators import api_view
from rest_framework.response import Response

import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = [
    "gemini-2.0-flash",           # Latest + fastest
    "gemini-1.5-flash",           # Fast fallback  
    "gemini-1.5-flash-8b",        # Lightest fallback — 1.5-pro ki jagah
]

ERROR_MSG = "Sorry, I couldn't reach the server. Make sure the backend is running."

@api_view(['POST'])
def chat(request):
    data = request.data
    history = data.get('history', [])
    message = data.get('message', '')

    contents = []
    for msg in history:
        # id field ignore karo, aur error messages skip karo
        if msg.get('content') == ERROR_MSG:
            continue
        role = 'user' if msg['role'] == 'user' else 'model'
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['content'])]))
    
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    # Gemini ko user/model alternating chahiye — last message user hona chahiye
    # Agar do consecutive same-role messages hain to fix karo
    cleaned = []
    for c in contents:
        if cleaned and cleaned[-1].role == c.role:
            # Same role consecutive — skip karo purana, naya rakho
            cleaned[-1] = c
        else:
            cleaned.append(c)

    last_error = None
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=cleaned,
            )
            return Response({"response": response.text, "model_used": model_name})
        except Exception as e:
            error_str = str(e).lower()
            if any(k in error_str for k in ["quota", "rate", "429", "resource_exhausted"]):
                last_error = e
                continue
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

    return Response({"error": f"All models rate limited. Last error: {str(last_error)}"}, status=429)