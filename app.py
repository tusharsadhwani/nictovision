import flask
from flask import request
# import draw_inference

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/upload', methods=['POST'])
def upload_img():
    img = request.files.get('upload_img', '')
    with open('uploaded_img.png', 'wb') as f:
        f.write(img.read())
    
    # new_path = draw_inference.infrence('uploaded_img', 'ios')
    # print(new_path)
    return "ok"