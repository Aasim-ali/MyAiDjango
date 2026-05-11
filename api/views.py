import httpx
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from dotenv import load_dotenv

load_dotenv()

ERROR_MSG = "Sorry, I couldn't reach the server. Make sure the backend is running."

@api_view(['POST'])
def chat(request):
    data = request.data
    history = data.get('history', [])
    message = data.get('message', '')

    messages = []
    for msg in history:
        if msg.get('content') == ERROR_MSG:
            continue
        role = "assistant" if msg['role'] == 'model' else "user"
        messages.append({"role": role, "content": msg['content']})
    
    messages.append({"role": "user", "content": message})

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-4-maverick:free",
                "messages": messages
            },
            timeout=30
        )
        result = response.json()
        if 'error' in result:
            print("OpenRouter error:", result['error'])
            return Response({"error": "AI service error."}, status=500)
        
        return Response({
            "response": result['choices'][0]['message']['content'],
            "model_used": "Llama 4 Maverick"
        })

        return Response({
            "response": result['choices'][0]['message']['content'],
            "model_used": "Llama 4 Maverick"
        })
    except Exception as e:
        print(str(e))
        return Response({"error": "Internal server error."}, status=500)