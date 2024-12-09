import requests
import json

def get_deepseek_response(system_prompt, user_prompt, api_key):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        return content
    else:
        response.raise_for_status()

def load_api_keys(file_path):
    with open(file_path, 'r') as file:
        keys = [line.strip() for line in file if line.strip()]
    return keys

if __name__ == "__main__":
    api_keys = load_api_keys('./config/key.txt')

    api_key = api_keys[0]  
    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello!"
    response = get_deepseek_response(system_prompt, user_prompt, api_key)
    print(response)
