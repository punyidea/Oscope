from scipy.signal import hilbert,butter,lfilter
import numpy as np


def circle_transform(x,fs, mod_fact=0.05, mod_freq=10):
    x_mono = np.mean(x,axis= 1)
    [X, env] = compress_and_normal(x_mono, 0, 3, 40 / fs * 2)
    H = hilbert(X)
    x_hilb = np.real(H)
    y_hilb = np.imag(H)

    [X, Y] = add_lowfreq_scale([x_hilb, y_hilb], fs, mod_fact, mod_freq)

    return X/np.max(np.abs(X)),Y/np.max(np.abs(Y))


def compress_and_normal(x,comp_fact,N,low_cutoff):
    x = x / max(x)
    [b, a] = butter(N, low_cutoff, 'low')
    env = np.abs(lfilter(b, a, np.abs(x))) + 1e-10
    h = env ** comp_fact
    comp_sig = x * h
    comp_sig = comp_sig/  np.max(np.abs(comp_sig))
    env_comp = abs(lfilter(b, a, np.abs(comp_sig))) + 1e-10
    return comp_sig, env_comp


def add_lowfreq_scale(x,fs,mod_fact,mod_freq):
    x = np.squeeze(x)/np.max(x)

    Y = np.cos(np.arange(np.size(x,1))/fs*2*np.pi*mod_freq) \
        + x[1,:]*mod_fact
    X = np.sin(np.arange(np.size(x,1)) / fs * 2 * np.pi * mod_freq) \
        + x[0,:]*mod_fact
    return X,Y
