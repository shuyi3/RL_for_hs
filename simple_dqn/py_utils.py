import json
import h5py
import numpy as np
np.set_printoptions(linewidth=300)

def py_help(fn_name):
    print help(fn_name)

def format_hierachy(json_string):
    return json.dumps(json_string, sort_keys = True, indent = 4)

def py_print(s):
    print s

def py_print_len(s):
    print len(s)

def dump_h5file(file_name, names, data):
    f = h5py.File(file_name + ".hdf5", "w")
    for i in xrange(len(names)):
        f.create_dataset(names[i], data=data[i])
    f.close()