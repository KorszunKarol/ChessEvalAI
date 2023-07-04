import psutil

# Get the number of system-level threads
num_threads = psutil.cpu_count(logical=True)

print("Number of threads:", num_threads)
