

from flask_appbuilder.api import BaseApi, expose
from . import appbuilder




class CeleryWorker(BaseApi):
    route_base = ''
    @expose('/worker/test')
    def greeting(self):
        return self.response(200, message="worker")


appbuilder.add_api(CeleryWorker)


