'''
Created on 17 Aug 2017

@author: mozat
'''
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

class HierachySoftmaxLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.coarse_weight = params['coarse_weight']
        self.fine_weight = params['fine_weight']
        self.class_map = self.get_class_map(params.get('class_map_name', 'labels_color.csv')) 
        if len(bottom) != 2:
            raise Exception("Wrong number of bottom blobs(prediction and label)")
        pass
    
    def get_class_map(self, class_map_name):
        class_mapping = {}
        with open(class_map_name, 'r') as fp:
            coarse_name = set()
            lines = fp.readlines()
            for line in lines:
                coarse_name.add(line.strip().split(',')[1].split('_')[2])
            for line in lines:
                class_mapping[int(line.strip().split(',')[0])] = list(coarse_name).index(line.strip().split(',')[1].split('_')[2])
        return class_mapping
    
    #bottom, top are protobuf message
    def reshape(self, bottom, top): 
        #num is the data number
        #count is the total element number
        if bottom[0].num != bottom[1].num: 
            raise Exception("Inputs must have the same dimension.")
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32) 
        top[0].reshape(1)
        pass
    
    def forward(self, bottom, top):
        #bottom[0] 
        scores = bottom[0].data - np.reshape(np.max(bottom[0].data, axis=1), (bottom[0].num, -1))
        exp_scores = np.exp(scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.predict_id = np.argmax(self.probs, axis=1)
        self.compute_weights = np.zeros((bottom[1].num, 1), dtype=np.float32)
        for idx in range(bottom[0].num):
            if self.predict_id[idx] == bottom[1].data[idx]: #correctly mapping  in fine grained
                self.compute_weights[idx, 0] = 1
            elif self.class_map[self.predict_id[idx]] == self.class_map[bottom[1].data[idx]]:
                self.compute_weights[idx, 0] = self.fine_weight
            else:
                self.compute_weights[idx, 0] = self.coarse_weight
        self.probs = self.probs*self.compute_weights            
        correct_logprobs = -np.log(self.probs[range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16)]) #label 
        loss = np.sum(correct_logprobs)/bottom[0].num
        self.diff[...] = self.probs
        top[0].data[...] = loss
        pass
    
    def backward(self, top, propagate_down, bottom):
        delta = self.diff
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                delta[range(bottom[0].num), np.array(bottom[1].data, dtype=np.uint16)] -= self.compute_weights[:,0]
            bottom[i].diff[...] = delta/bottom[0].num
        pass

def train_net():
    solver = "models/bvlc_googlenet_multask/solver.prototxt"
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    solver_config = caffe_pb2.SolverParameter()
    text_format.Merge(open(solver, 'r').read(), solver_config)
    max_iter = solver_config.max_iter
    solver = caffe.SGDSolver(solver)
    for _ in range(max_iter):
        solver.step(1)

if __name__ == '__main__':
    train_net()
    