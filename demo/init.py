from flask import Flask, render_template
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, 'static')
templates_dir = os.path.join(BASE_DIR, 'templates')

def create_app():
    app = Flask(__name__, template_folder='template')
    return app

