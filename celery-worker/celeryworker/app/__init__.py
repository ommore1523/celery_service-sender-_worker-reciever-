import logging

from flask import Flask
from flask_appbuilder import AppBuilder, SQLA

"""
 Logging configuration
"""

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)
app.config.from_object("config")
db = SQLA(app)
appbuilder = AppBuilder(app, db.session)



REDIS_HOST='127.0.0.1'
REDIS_PORT=6379

redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

from celery import Celery
celeryapp = Celery('worker', broker=redis_uri, backend=redis_uri,)

from . import views
