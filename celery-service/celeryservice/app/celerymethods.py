from os import scandir

import celery
from . import celerymethods
from . import celeryapp

def check_celery_isUp(queueName):
    allQueues = celeryapp.control.inspect().active_queues()
    if allQueues is not None:
        for key in allQueues.keys():
            for item in allQueues[key]:
                if item['name'] == queueName:
                    return True      
    else:
        return False
        


def schedular(lst):

    celeryapp.send_task('run.add_numbers_master', queue = "demo", args=([lst]))

def soln():
    isAvail = check_celery_isUp("demo")
    if isAvail:
        task_id = celeryapp.send_task('run.solve', queue = "demo")
        print("task_id",task_id)
    else:
        return "celeryDown"

def revoketask(id):
    # pass
    resp = celeryapp.control.revoke(id, terminate=True)
    print(resp)
