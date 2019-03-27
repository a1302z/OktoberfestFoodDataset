"""
Helpers for sending and receiving numpy arrays
"""

import numpy as np


def recvall(sock, count):
    """
    sock: the socket which receives the data
    count: the size of the buffer that was sent
    returns:
    """
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def send_np_array(s, a):
    """
    s: the socket
    a: the array that should be send
    """
    string_data = a.tostring()

    # first send the size of the encoded array and its shape so that the receiver can reconstruct the array
    m = np.zeros(8, dtype=np.int32)
    m[0] = len(string_data)
    m[1:len(a.shape)+1] = a.shape

    s.sendall(m.tostring())
    s.sendall(string_data)


def receive_np_array(s, dtype):
    """
    s: the socket
    dtype: the type of the array
    returns: the received array
    """
    # first receive the size of the buffer and the shape of the array
    m = np.frombuffer(recvall(s, 8*4), dtype=np.int32)
    shape = m[1:]
    shape = shape[shape != 0]
    # receive the array, cast it to dtype and then reshape it accordingly
    a = np.frombuffer(recvall(s, m[0]), dtype=dtype).reshape(shape)
    return a
