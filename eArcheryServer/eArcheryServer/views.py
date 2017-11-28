from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
import numpy as np
from numpy import genfromtxt, savetxt
# from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from sklearn.ensemble import RandomForestClassifier
import pickle
import os.path
import os
import csv

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
training_csv_filename = os.path.join(__location__, 'processed.csv')
pickle_filename = 'model.pkl'
def home(request):
    raw = request.GET.get("raw","")
    raw = raw.split("!")
    p = getLabelForData(raw) if raw != "" else "No data received"
    return HttpResponse(labelForIndex(int(p)))

#Helper methods here on output
def getLabelForData(accelerometer_raw):
    accelerometer_vector = preprocess(accelerometer_raw)

    if (not os.path.isfile(pickle_filename)):
        print "Training model"
        #https://www.kaggle.com/c/digit-recognizer/forums/t/2299/getting-started-python-sample-code-random-forest
        # dataset = genfromtxt(open(training_csv_filename), delimiter=',', dtype='f8')[1:]
        dataset = loadDataset(training_csv_filename)
        target = []
        train = []
        for item in dataset:
            target.append(item[-1])
            train.append(item[:-1])
        print target
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(train, target)
        pickle_dump(pickle_filename, rf)

    print "Predicting on model using vector: ", accelerometer_vector
    rf = pickle_load(pickle_filename)
    prediction = rf.predict(accelerometer_vector[:-1])
    print "Prediction is :: ", prediction
    return prediction


def preprocess(accelerometer_raw):
    #List of accelerometer samples, each like '18341: 11244,2448,11012*'

    x_stream = []
    y_stream = []
    z_stream = []
    processed_stream = [0] * 10

    #process raw data into streams
    for sample in accelerometer_raw:
        if len(sample.split(':')) == 2:
            timestamp, acc_data = sample.split(':')
            timestamp = float(timestamp)
            if timestamp < 100000000.0:
                if len(acc_data.split(',')) == 3:
                    x,y,z = acc_data.split(',')
                    x=x.replace("*","")
                    x=x.replace(" ","")
                    y=y.replace("*","")
                    y=y.replace(" ","")
                    z=z.replace("*","")
                    z=z.replace(" ","")
                    x_stream.append(float(x))
                    y_stream.append(float(y))
                    z_stream.append(float(z))

    x_stream = np.array(x_stream)
    y_stream = np.array(y_stream)
    z_stream = np.array(z_stream)

    #For each stream, filter, bin, etc
    base = 0 #to set x,y,z, etc
    additive = 0 #to set x,y,z, etc
    for stream in [x_stream, y_stream, z_stream]:
        # print "Stream: Original ", stream
        stream_avg  = smooth(stream,3)
        # print "Stream: Average  ", stream_avg
        stream_sg   = savitzky_golay(stream_avg, 7, 3) #Degree 3 and window length 7
        # print "Stream: SG       ",stream_sg
        stream_bins = binned_statistic(stream_sg, stream_sg, bins=5)[0]
        # print "Stream: bins     ",len(stream_bins)
        for bn in stream_bins[2:]:
            if np.isnan(bn):
                bn = 0.0
            processed_stream[base + additive] = bn
            additive += 3
        base += 1
        additive = 0

    #Returns centroids in form x1,y1,z1,x2,y2,z2,x3,y3,z3,?
    return processed_stream

def loadDataset(filename):
    data = []
    row0 = True
    with open(filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile.read().splitlines())
            for row in spamreader:
                if row0:
                    row0 = False
                else:
                    floatrow = []
                    for item in row:
                        if indexForLabel(item.replace("'","")) == -1:
                            floatrow.append(float(item))
                        else:
                            floatrow.append(indexForLabel(item.replace("'","")))
                    data.append(floatrow)
    return np.array(data)

def labelForIndex(index):
    if index == 0: return 'good'
    if index == 1: return 'pluck'
    if index == 2: return 'dead'

def indexForLabel(label):
    if label == 'good': return 0
    if label == "pluck": return 1
    if label == 'dead': return 2
    return -1

#http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

#http:#scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def pickle_load(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

def pickle_dump(filename, data):
    print "Dumping pickle"
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()
