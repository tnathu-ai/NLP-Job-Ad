import os
from flaskblog.errors.handlers import errors
from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(16)
app.register_blueprint(errors)

from flaskblog import routes