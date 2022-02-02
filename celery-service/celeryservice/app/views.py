from flask_appbuilder.api import BaseApi, expose
from . import appbuilder
from . import app
from . import celerymethods
from . import celeryapp
from celery import Signature,chord,Celery
''' ----------------------------------------------------------------------------------
                        Celery Implementation
    -----------------------------------------------------------------------------------'''

REDIS_HOST='127.0.0.1'
REDIS_PORT=6379



class CeleryService(BaseApi):
    route_base = '/service'
    @expose('/test')
    def greeting(self):
        lst = [2,3]
        total = [[1,2],[3,4],[5,6]]
        suffix = ["first","second","third"]
        # ** Objective ** : 3 + 7 + 11 =  21
        signatureList = []
        for lst in range(len(total)):
            # celeryapp.send_task('run.add_numbers', queue = "demo", args=(total[lst],))
            sig = suffix[lst]
            sig = Signature("run.add_numbers" ,  queue = "demo", args=(total[lst],) )
            signatureList.append(sig)


        redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
        celeryapp = Celery('nlp', broker=redis_uri, backend=redis_uri)


        callbk = Signature("run.gettotol",queue = "demo")

        res = chord(signatureList)(callbk)

        return self.response(200, message="Hello")



appbuilder.add_api(CeleryService)
