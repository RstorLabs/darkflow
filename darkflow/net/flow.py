import os
import shutil
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
import requests
import uuid
import pathlib
import cv2
import io
import socket    #for sockets
import sys    #for exit

# from fum import fum_yield

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def discover_host():
    # create dgram udp socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except socket.error:
        print ('Failed to create socket')
        sys.exit()

    host = '255.255.255.255';
    port = 8789;

    while(1) :
        msg = "hello"
        msg = msg.encode()
        try :
            #Set the whole string
            s.sendto(msg, (host, port))
            
            # receive data from client (data, addr)
            d = s.recvfrom(1024)
            reply = d[0]
            addr = d[1]
            
            print ('Server reply : ' + str(reply))
            print ('IP address : ' + str(addr[0]) +' port : '+str(addr[1]))
            if reply == b'I am a Coltrane Demo Host':
                return addr
        except (socket.error, msg):
            print ('Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    try:
        addr = os.environ["HOST_IP"]
    except:
        addr = discover_host()
        addr = addr[0]
    url = "http://"+addr+":1337"
    randomizer = str(uuid.uuid4())
    inp_path = os.path.join(self.FLAGS.imgdir,randomizer)
    pathlib.Path(inp_path).mkdir()


    while True:
        # next line hard-coded because using for demo 
        # and also don't want to delete things willy-nilly
        folder = '/tmp/input'
        

        r = requests.get(url)
        if r.status_code != 200:
            time.sleep(.125)
            continue


        # im_cv2 = io.BytesIO(r.content)
        # im_bytes = bytearray(im_cv2.read())
        np_arr = np.fromstring(r.content, np.uint8)
        im_cv2 = cv2.imdecode(np_arr, 1)

        inp_feed = [np.expand_dims(self.framework.preprocess(im_cv2), 0)]

        #     # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
        # print(out[0])
        output = out[0]

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        result = self.framework.postprocess(output, im_cv2, save = False)
        # pool.map(lambda p: (lambda i, prediction:
        #     self.framework.postprocess(
        #         prediction, os.path.join(inp_path, this_batch[i])))(*p),
        #     enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
        jframe = cv2.imencode(".jpg", result)
        j_img = io.BytesIO(jframe[1])
        jbytes = bytearray(j_img.read())
        requests.post(url, jbytes)
        
        # time.sleep(.125)
