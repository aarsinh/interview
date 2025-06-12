import requests
import json

response = requests.post("http://localhost:8000/analyze_video", params={"filename": "20250611_233519_new.mp4"})

with open("results.json", "w") as file:
    json.dump(response.json(), file, indent=2)