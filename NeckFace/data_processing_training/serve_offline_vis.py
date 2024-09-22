'''
Serve a prediction file over TCP for offline visualization
7/20/2023, Ruidong Zhang, rz379@cornell.edu
'''
import socket
import argparse
import numpy as np
from time import sleep

def serve_pred(pred_path, visualization_port=6001, visualization_addr='127.0.0.1'):

    vis_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    vis_sock.connect((visualization_addr, visualization_port))

    print('Streaming visualization to %s:%d' % (visualization_addr, visualization_port))

    preds = np.load(pred_path)
    t0 = preds[0][0]

    try:
        for p in preds:
            sleep(p[0] - t0)
            vis_array = ','.join([str(x) for x in p]) + '\n'
            # print(preds)
            vis_sock.sendall(vis_array.encode())
            t0 = p[0]
    except Exception as e:
        print(e)
    vis_sock.close()
    print('Streaming done.')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser('NeckFace offline visualization - Laptop')
    parser.add_argument('-p', '--pred', help='path to the prediction file')
    parser.add_argument('--port', help='Port where the visualization program is listening on', type=int, default=6001)
    parser.add_argument('--addr', help='Address where the visualization program is listening on, default localhost', default='127.0.0.1')

    args = parser.parse_args()

    serve_pred(args.pred, args.port, args.addr)