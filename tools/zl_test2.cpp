//
// Created by mozat on 14/8/17.
//
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace std;
using namespace cv;

static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
    return lhs.first > rhs.first;
}

static vector<int> Argmax(const vector<float> &v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}


class Classifier {
public:
    Classifier(const string &model_file, const string &trained_file, const string &mean_file, const string &label_file);

    vector<pair<string, float> > Classify(const Mat &img, int N = 5);

private:
    void SetMean(const string &mean_file);

    vector<float> Predict(const Mat &img);

    void Preprocess(const Mat &img, vector<Mat> *input_channels);

    void WrapInputLayer(vector<Mat> *input_channels);

private:
    shared_ptr<Net<float> > net_;
    Size input_geometry_;
    int num_channels_;
    Mat mean_;
    vector<string> labels_;
};

Classifier::Classifier(const string &model_file, const string &trained_file,
                       const string &mean_file,
                       const string &label_file) {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    SetMean(mean_file);
    std::ifstream labels(label_file.c_str());
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));
    Blob<float> *output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension";
}

vector<pair<string, float> > Classifier::Classify(const cv::Mat &img, int N) {
    std::vector<float> output = Predict(img);
    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output, N);
    std::vector<pair<string, float> > predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }
    return predictions;
}


void Classifier::SetMean(const string &mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    std::vector<cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }
    Mat mean;
    merge(channels, mean);
    Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

vector<float> Classifier::Predict(const Mat &img) {
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    vector<Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);
    net_->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    return vector<float>(begin, end);
}

void Classifier::WrapInputLayer(vector<Mat> *input_channels_) {
    Blob<float> *input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels_->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat &img, vector<Mat> *input_channels) {
    Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
    cv::split(sample_normalized, *input_channels);
}


int main() {
    string model_file = "/home/mozat/Desktop/test_net/cifar10_quick_deploy.prototxt";
    string trained_file = "/home/mozat/Desktop/test_net/cifar10_quick_iter_5000.caffemodel";
    string mean_file = "/home/mozat/Desktop/test_net/mean.binaryproto";
    string label_file = "/home/mozat/Desktop/test_net/labels.txt";
    cv::Mat img = cv::imread("/home/mozat/Desktop/test_net/1.png");
    CHECK(!img.empty()) << "unable to decode image";
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    std::vector<pair<string, float> > predictions = classifier.Classify(img);
    for (size_t i = 0; i < predictions.size(); ++i) {
        pair<string, float> p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
    }
}
