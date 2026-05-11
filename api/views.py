import httpx
import os
import time
from rest_framework.decorators import api_view
from rest_framework.response import Response
from dotenv import load_dotenv

load_dotenv()

models = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
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

    start_time = time.time()

    for model in models:
        elapsed = time.time() - start_time
        remaining = 22 - elapsed

        if remaining < 3:
            print(f"Time limit reached ({elapsed:.1f}s elapsed), stopping failover.")
            break

        per_model_timeout = httpx.Timeout(
            connect=2.0,
            read=min(8.0, remaining - 1),
            write=2.0,
            pool=2.0
        )

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={"model": model, "messages": messages},
                timeout=per_model_timeout
            )

            if response.status_code != 200:
                print(f"{model} returned status {response.status_code}")
                try:
                    err_data = response.json()
                    if 'error' in err_data:
                        print(f"{model} error details:", err_data['error'])
                except Exception:
                    pass
                continue

            result = response.json()

            if 'error' in result:
                print(f"{model} error:", result['error'])
                continue

            return Response({
                "response": result['choices'][0]['message']['content'],
                "model_used": model
            })

        except httpx.TimeoutException:
            print(f"{model} timed out after {time.time() - start_time:.1f}s total")
            continue
        except httpx.RequestError as e:
            print(f"{model} request error: {e}")
            continue
        except Exception as e:
            print(f"{model} unexpected error: {e}")
            continue

    return Response(
        {"error": "All models are busy. Please try again."},
        status=503
    )