from flask import Flask
from configure_endpoints import *
from ml_backend import *
app = Flask(__name__)
model = create_model()

configure_app(app, model)

if __name__ == '__main__':
    app.run()
