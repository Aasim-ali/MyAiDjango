import httpx
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from dotenv import load_dotenv

load_dotenv()

models = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "inclusionai/ring-2.6-1t:free",
    "baidu/cobuddy:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free"
]

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

    for model in models:
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={"model": model, "messages": messages},
                timeout=30
            )
            result = response.json()

            if 'error' in result:
                print(f"{model} error:", result['error'])
                continue  # agli model try karo

            return Response({
                "response": result['choices'][0]['message']['content'],
                "model_used": model
            })
        except httpx.TimeoutException:
            continue
        except Exception as e:
            print(str(e))
            continue

    return Response({"error": "All models are busy. Please try again."}, status=503)
