import flask
from flask import request
from flask_cors import CORS
import draw_inference

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/', methods=['GET'])
def homepage():
    with open('index.html') as f:
        return f.read()

@app.route('/', methods=['POST'])
def upload_img():
    print(request.files)
    img = request.files.get('upload_img', None)
    with open('uploaded_img.dng', 'wb') as f:
        f.write(img.read())
    
    new_path = draw_inference.infrence('./uploaded_img.dng', 'ios')
    print(new_path)
    return "ok"