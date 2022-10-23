import os
from flaskjob.errors.handlers import errors
from flask import Flask

app = Flask(__name__)
app.register_blueprint(errors)
app.config['SECRET_KEY'] = os.urandom(16)


from flaskjob import routes