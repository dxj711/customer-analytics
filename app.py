from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder where uploaded files will be stored
#os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('home.html')

@app.route('/run_churn_prediction', methods=['POST'])
def run_churn_prediction():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'churn_prediction.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/run_customer_segmentation', methods=['POST'])
def run_customer_segmentation():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'customer_segmentation.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/run_customerltv', methods=['POST'])
def run_customerltv():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'CustomerLTV.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/run_Chat_audio', methods=['POST'])
def run_Chat_audio():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'Chat_audio.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    

@app.route('/run_market_response_mod', methods=['POST'])
def run_market_response_mod():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'market_response_mod.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    

@app.route('/run_nextpurchase', methods=['POST'])
def run_nextpurchase():
    try:
         
        subprocess.Popen(['streamlit', 'run', 'nextpurchase.py'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
