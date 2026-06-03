import signal
import time

print("Hi")

def handler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

# do nothing
print("Going to sleep...")
time.sleep(100000)
