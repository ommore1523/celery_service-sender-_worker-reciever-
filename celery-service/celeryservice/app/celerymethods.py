from os import scandir

import celery
from . import celerymethods
from . import celeryapp



def schedular(lst):
    
    celeryapp.send_task('run.add_numbers_master', queue = "demo", args=([lst]))

def soln():
    celeryapp.send_task('run.solve', queue = "demo")