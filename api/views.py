import httpx
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from dotenv import load_dotenv

load_dotenv()

models = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ring-2.6-1t:free",
    "baidu/cobuddy:free",
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
                timeout=8.0  # Reduced from 30 to prevent Gunicorn worker timeout
            )
            
            # Handle non-200 responses gracefully
            if response.status_code != 200:
                print(f"{model} returned status {response.status_code}")
                try:
                    err_data = response.json()
                    if 'error' in err_data:
                        print(f"{model} error details:", err_data['error'])
                except Exception:
                    print(f"{model} failed to parse error JSON")
                continue

            result = response.json()

            if 'error' in result:
                print(f"{model} error:", result['error'])
                continue  # Try next model

            return Response({
                "response": result['choices'][0]['message']['content'],
                "model_used": model
            })
            
        except httpx.TimeoutException:
            print(f"{model} timed out")
            continue
        except httpx.RequestError as e:
            print(f"{model} request error:", str(e))
            continue
        except Exception as e:
            print(f"{model} unexpected error:", str(e))
            continue

    return Response({"error": "All models are busy. Please try again."}, status=503)
