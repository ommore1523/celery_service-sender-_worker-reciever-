
from flask_appbuilder.api import BaseApi, expose
from . import appbuilder
from . import app
from flask import request
from . import celerymethods

''' ----------------------------------------------------------------------------------
                        Celery Implementation
    -----------------------------------------------------------------------------------'''


"""



"""


class CeleryService(BaseApi):
    route_base = '/service'


    @expose('/schedulartest', methods=['POST'])
    def greeting1(self):
        celerymethods.schedular(request.json['number_list'])

        return self.response(200, message="Hello")
        
    @expose('/solver')
    def greeting2(self):
        celerymethods.soln()

        return self.response(200, message="Hello")



appbuilder.add_api(CeleryService)


"""
1. in_ts_data ==> 2nd function
2. 

"""