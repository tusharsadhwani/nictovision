import flask
from flask import request, redirect
from flask_cors import CORS
import draw_inference

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/', methods=['GET'])
def homepage():
    with open('index.html') as f:
        return f.read()

@app.route('/result/ios.png', methods=['GET'])
def result():
    if os.path.exists('result/ios.png'):
        with open('result/ios.png', 'rb') as f:
            return f.read()
    
    return ""

@app.route('/original/input.png', methods=['GET'])
def orig():
    if os.path.exists('original/input.png'):
        with open('original/input.png', 'rb') as f:
            return f.read()
    
    return ""

@app.route('/', methods=['POST'])
def upload_img():
    print(request.files)
    img = request.files.get('upload_img', None)
    with open('uploaded_img.dng', 'wb') as f:
        f.write(img.read())
    
    new_path = draw_inference.infrence('./uploaded_img.dng', 'ios')
    print(new_path)
    return redirect('/')