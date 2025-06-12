First create a venv and install the required packages
```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on windows
pip install -r requirements.txt
```

To run the app, first run `python main.py` in the root directory.

Then run the following to run the detect API:
```
cd detect-app
python app.py
```
Open the index.html file in browser and upload a file to analyze
Uploaded files will go to uploads/ directory.

Note: This will require you to run a Roboflow Inference server and get an API key from Roboflow. Add this API Key to a .env file 
in the following format: API_KEY=YOUR_API_KEY
