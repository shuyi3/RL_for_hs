import zmq
import time
import sys
from cnn_model import *

port = "5556"
if len(sys.argv) > 1:
    port = sys.argv[1]
    int(port)

fcn = compile_fcn_model()

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

# global_ft = np.zeros((1,26))
# board_ft = np.zeros((1, 9, 17, 5))
# hand_ft = np.zeros((1, 19, 23))
global_ft = np.zeros((1, 26), dtype=np.float32)
board_ft = np.zeros((1, 9 * 17 * 5), dtype=np.float32)
hand_ft = np.zeros((1, 18 * 23), dtype=np.float32)
play_ft = np.zeros((1, 23), dtype=np.float32)
# X=[global_ft, board_ft, hand_ft]
X=[global_ft, board_ft, hand_ft, play_ft]
# res = np.argmax(cnn.predict(X)[0])
# print res
count = 0

while True:
    #  Wait for next request from client
    frm = socket.recv(copy=False)
    # print type(frm.bytes), frm.bytes
    val = map(int, frm.bytes.split(','))
    for i in xrange(100): X[0][0] = 1
    res = np.argmax(fcn.predict(X)[0])
    # time.sleep (1)
    print 'pred:', res
    socket.send(res)
    count += 1
    if count == 1: start = time.time()
    if count == 3001: break

print 'time passed:', time.time() - start

start = time.time()
for i in xrange(3000):
    res = np.argmax(fcn.predict(X)[0])

print 'time passed:', time.time() - start
