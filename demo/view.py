from flask import Flask, render_template, request
import os
import init
from predict import predict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, 'static')
templates_dir = os.path.join(BASE_DIR, 'Web/templates')

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/mainPage')
def mainPage():
    return render_template('mainPage.html')

@app.route('/modelReview', methods=['GET', 'POST'])
def modelReview():
    return render_template('modelReview.html')

@app.route('/resultShow', methods=['GET', 'POST'])
def resultShow():
    return  render_template('resultShow.html')

@app.route('/demoPage', methods=['GET', 'POST'])
def demoPage():
    return render_template('demoPage.html')

@app.route('/mainPage', methods=["POST"])
def some_function():
    text = request.form.get('textbox')
    t1, t2 = predict(text)
    return render_template('mainPage.html', mes1=t1, mes2=t2)

if __name__ == '__main__':
    app.run(debug=True)