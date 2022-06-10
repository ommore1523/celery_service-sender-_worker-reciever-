from celery import Signature,chord,Celery

REDIS_HOST='192.168.100.178'
REDIS_PORT=11379

redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
celeryapp = Celery('xpnp', broker=redis_uri, backend=redis_uri)

i = celeryapp.control.inspect().scheduled()

print(i)