#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  virtual inline bool AutoTopBlobs() const { return true; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};



/**
 * @brief A weighted version of SoftmaxWithLossLayer.
 *
 * TODO: Add description. Add the formulation in math.
 */

template <typename Dtype>
class WeightedSoftmaxWithLossLayer:public LossLayer<Dtype> {
public:
  explicit WeightedSoftmaxWithLossLayer(const LayerParameter& param): LossLayer<Dtype>(param){

  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom, 
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);

  virtual inline const char* type() const{
    return "WeightedSoftmaxWithLoss";
  }
  virtual inline int ExactNumBottomBlobs() const{
    return -1;
  }
  virtual inline int MinBottomBlobs() const{
    return 1;
  }
  virtual inline int MaxBottomBlobs() const{
    return 2;
  }
  virtual inline int ExactNumTopBlobs() const{
    return -1;
  }
  virtual inline int MinTopBlobs() const{
    return 1;
  }
  virtual inline int MaxTopBlobs() const{
    return 2;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  /// The label indicating that an instance should be ignored.
  bool has_ignore_label_;
  int ignore_label_;
  /// Whether to normalize the loss by the total number of values present
  ///(otherwise just by the batch size).
  bool normalize_;
  int softmax_axis_, outer_num_, inner_num_;

  //label that we want to increase weight
  vector<float> pos_mult_;
    vector<int> pos_cid_;
    map<int, float> weight_map_;
    Dtype *weights_;
};
}  // namespace caffe
#endif  // CAFFE_LOSS_LAYER_HPP_