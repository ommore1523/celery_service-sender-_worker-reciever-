from . import views  
import time 
from . import celeryapp

@celeryapp.task
def Hello():
    time.sleep(5)
    print("Hi I am running without que name")