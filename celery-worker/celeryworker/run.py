from app import celeryapp
import time



def hello():
    time.sleep(4)
    print("Hello")

# @celeryapp.task
# def multiprocessfunction():
#     for i in range(10):
#         time.sleep(1)
#         print("hello")
        # process = multiprocessing.Process(target=hello,args=(2,))
        # process.start()

# task within  task with same queue and worker
@celeryapp.task
def add_numbers(lst):
    print("add_numbers function",lst)
    # celeryapp.send_task('run.multiprocessfunction',queue = 'demotest')
    return sum(lst)

@celeryapp.task
def gettotol(data):
    print("gettotol function",sum(data))



