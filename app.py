import flask
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True
from urllib.parse import urlparse
from email.parser import BytesParser

@app.route('/upload', methods=['POST'])
def upload_img():
    img = request.files.get('upload_img', '')
    with open('uploaded_img.png', 'wb') as f:
        f.write(img.read())
    return "ok"