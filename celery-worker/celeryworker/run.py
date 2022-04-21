

import celery
from app import celeryapp
import time
import requests
import json

from celery import Signature,chord,Celery
import string
import random
# import forecasting



# @celeryapp.task
# def add_numbers(lst,schedular):
#     if schedular:
#         print("schedular",schedular)
#         # { "number_list": [[1,2],[2,3],[3,4]] }
#         data = json.dumps({'number_list': lst})

#         response = requests.post('http://127.0.0.1:8080/service/test', data=data, headers= {'Content-type': 'application/json', 'Accept': 'text/plain'})
#         if response.status_code != 200:
#             raise ("SCHEDULARFAILED")

#     else:
#         print("service", schedular)
#         print("add_numbers function",lst,schedular)
#         # celeryapp.send_task('run.multiprocessfunction',queue = 'demotest')
#         return sum(lst)


@celeryapp.task
def gettotol(*args, **kwargs):
    print("get total functions",args,kwargs)





REDIS_HOST='127.0.0.1'
REDIS_PORT=6379

def add_task(list_n):
    suffix = [''.join(random.choices(string.ascii_uppercase + string.digits, k = 4)) for i in range(len(list_n))]
    # ** Objective ** : 3 + 7 + 11 =  21
    signatureList = []
    for lst in range(len(list_n)):
        # celeryapp.send_task('run.add_numbers', queue = "demo", args=(total[lst],))
        sig = suffix[lst]
        sig = Signature("run.add_numbers" ,  queue = "child", args=([list_n[lst]]) )
        signatureList.append(sig)

    redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    celeryapp = Celery('nlp', broker=redis_uri, backend=redis_uri)

    callbk = Signature("run.gettotol",queue = "child")

    res = chord(signatureList)(callbk)

@celeryapp.task
def add_numbers(lst):
    print("add_numbers",lst)
    return sum(lst)


@celeryapp.task
def add_numbers_master(lst):
    print(lst)
    add_task(lst)

@celeryapp.task
def solve():
    import time
    for i in range(10):
        print("solver started", i)
        time.sleep(2)
    # forecasting.rundart()
