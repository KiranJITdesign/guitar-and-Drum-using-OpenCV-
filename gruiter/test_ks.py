import numpy as np
import scipy.signal
import time

sample_rate = 44100
freq = 330.0 # High E
duration = 2.0
N = int(sample_rate * duration)

start = time.time()
L = int(sample_rate / freq)
b = np.array([1.0])
a = np.zeros(L + 2)
a[0] = 1.0
loss = 0.995
a[L] = -0.5 * loss
a[L+1] = -0.5 * loss

x = np.zeros(N)
x[:L] = np.random.uniform(-1, 1, L)
out = scipy.signal.lfilter(b, a, x)
end = time.time()

print(f"Time taken: {end - start:.5f} seconds")
print(f"out shape: {out.shape}, max: {np.max(out)}")
