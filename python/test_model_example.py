'''
Created on 4 Aug 2017

@author: mozat
'''

import sys
sys.path.append('/home/mozat/git/caffe/python')
import caffe
import numpy as np
import lmdb
from google.protobuf import text_format
from caffe.proto import caffe_pb2

def record_performence():
    #performence record
    #for all the wrong results
    #41.1 are within the same big category
    #58.9 are within different big category
    pass
def load_label(label_csv_file = './create_lmdb/labels_color.csv'):
    labels = {}
    with open(label_csv_file, 'r') as fp:
        for line in fp.readlines():
            labels[int(line.strip().split(',')[0])] = line.strip().split(',')[1]
    return labels
def get_dynamic_batch_size(new_batch_size, deploy_prototxt):
    net_param = caffe_pb2.NetParameter()
    text_format.Merge(open(deploy_prototxt, 'r').read(), net_param)
    net_param.layer[0].input_param.shape[0].dim[0] = new_batch_size
    new_deploy_file = deploy_prototxt+'.generated'
    with open(new_deploy_file, 'wb') as fp:
        text_format.PrintMessage(net_param, fp)
    return new_deploy_file
    
def test():
    import time
    labels = load_label()
    deploy_prototxt = "./models/bvlc_googlenet-modified/deploy-color.prototxt"
    weight_file = "./models/bvlc_googlenet-modified/snapshot/googlenet__iter_55000.caffemodel"
    test_lmdb = lmdb.open("./create_lmdb/test_color_lmdb")
    test_lmdb_txn = test_lmdb.begin()
    test_lmdb_cursor = test_lmdb_txn.cursor()
    batch_size = 300
    new_deploy_file = get_dynamic_batch_size(batch_size, deploy_prototxt)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(new_deploy_file, weight_file, caffe.TEST)
    #net = caffe.Net(new_deploy_file, caffe.TEST)
    #net.copy_from(weight_file)
    datum = caffe_pb2.Datum()
    total, main_acc, sub_acc = 0, 0, 0
    index = 0
    key_name = []
    ground_truth_label = np.zeros((batch_size, ), dtype=np.float32)
    start_time = time.time()
    for key, value in test_lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        key_name.append(key)
        ground_truth_label[index] = label
        img = caffe.io.datum_to_array(datum)
        img = img[:,1:225, 1:225].astype(np.float32)
        mean = np.array([126.88187408, 137.67976379 , 162.782653810])
        mean = mean[:, np.newaxis, np.newaxis]
        img -= mean
        net.blobs['data'].data[index] = img
        index = index + 1
        total = total + 1
        if index == batch_size:
            print total, time.time()-start_time
            out = net.forward()
            predict_labels = np.argmax(out['prob'], axis=1)
            for idx in range(batch_size):
                if ground_truth_label[idx] == predict_labels[idx]:
                    sub_acc = sub_acc+1
                    main_acc = main_acc+1
                else:
                    label_name = labels[ground_truth_label[idx]]
                    predict_label_name = labels[predict_labels[idx]]
                    if label_name.split('_')[2] == predict_label_name.split('_')[2]:
                        main_acc = main_acc+1
                    predict_log.info('{} {} {} {} {}'.format(
                                                    key_name[idx],
                                                    label_name, predict_label_name,
                                                    label_name.split('_')[2], predict_label_name.split('_')[2]
                                                    )
                                     )
            index = 0
            key_name = []
            
    if index!=0:
        out = net.forward()
        predict_labels = np.argmax(out['prob'], axis=1)
        for idx in range(index):
            if ground_truth_label[idx] == predict_labels[idx]:
                sub_acc = sub_acc+1
                main_acc = main_acc+1
            else:
                label_name = labels[ground_truth_label[idx]]
                predict_label_name = labels[predict_labels[idx]]
                if label_name.split('_')[2] == predict_label_name.split('_')[2]:
                    main_acc = main_acc+1
                predict_log.info('{} {} {} {} {}'.format(
                                                key_name[idx],
                                                label_name, predict_label_name,
                                                label_name.split('_')[2], predict_label_name.split('_')[2]
                                                )
                                 )
    predict_log.info('{} {} {}'.format(total, sub_acc*1.0/total, main_acc*1.0/total))
    print 'finished', time.time()-start_time, batch_size

if __name__ == '__main__':
    import logging
    if len(sys.argv)<=1:
        logging.basicConfig(level = logging.DEBUG,
                                          filename="predict_batch.log",
                                          format='%(asctime)s %(message)s',
                                          filemode='w'
                                          )
    else:
        logging.basicConfig(level = logging.DEBUG,
                                          filename=sys.argv[1],
                                          format='%(asctime)s %(message)s',
                                          filemode='w'
                                          )
    predict_log = logging.getLogger()
    test()