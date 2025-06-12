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
