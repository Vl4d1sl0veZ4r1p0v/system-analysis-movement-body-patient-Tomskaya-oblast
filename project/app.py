from flask import Flask, render_template, flash, request, redirect, url_for
from flask import request
import logging
import os
import model

BREAK_FOLDER = 'data'
ALLOWED_EXTENSIONS = ['csv']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = BREAK_FOLDER
desision_model = model.Model('RF1_model.pkl')
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api', methods=['POST'])
def upload_file():
    logging.info(request.files)
    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            logging.info('Start prediction')
            prediction = desision_model.predict(file)
            logging.info("Prediction complited: %s" % prediction)
            if prediction == 0:
                result = 'Болезни Паркинсона не обнаружено'
            else:
                result = 'Обнаружена болезнь Паркинсона'
            return result
        except Exception as e:
            logging.error("Something went wrong..", exc_info=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return e
    else:
        logging.warning("Wrong file: %s" % file.filename)
        return "Wrong file"

if __name__ == '__main__':
    logging.info("Start appication")
    app.run(port = 5000, debug = False)

