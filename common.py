import threading


stat_flags = {
    'start': False,
    'waiting': False,
    'quit': False,
    'done': False
}

print_lock = threading.Lock()
