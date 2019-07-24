#include "GMM.h"
#include "GrabCut.h"
#include "GraphCut.h"
#include <iostream>
#include <limits>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;
using namespace std;

// beta = 1/(2*avg(sqr(||color[i] - color[j]||))) 
static double cal_Beta(const Mat& _image) {
	double beta;
	double total_tmp = 0;
	for (int y = 0; y < _image.rows; y++) {
		for (int x = 0; x < _image.cols; x++) {
			Vec3d color = (Vec3d)_image.at<Vec3b>(y, x);
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_image.at<Vec3b>(y, x - 1);
				total_tmp += diff.dot(diff);
			}
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_image.at<Vec3b>(y - 1, x - 1);
				total_tmp += diff.dot(diff);
			}
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_image.at<Vec3b>(y - 1, x);
				total_tmp += diff.dot(diff);
			}
			if (y > 0 && x < _image.cols - 1) {
				Vec3d diff = color - (Vec3d)_image.at<Vec3b>(y - 1, x + 1);
				total_tmp += diff.dot(diff);
			}
		}
	}
	total_tmp *= 2;
	if (total_tmp <= std::numeric_limits<double>::epsilon()) beta = 0;
	else beta = 1.0 / (2 * total_tmp / (8 * _image.cols * _image.rows - 6 * _image.cols - 6 * _image.rows + 4));
	return beta;
}
// �����������ص�Ȩ��
static void cal_weight(const Mat& _image, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	// ��i��j�Ǵ�ֱ����ˮƽ��ϵʱ��dis(i, j) = 1�����ǶԽǹ�ϵʱ��dis(i, j) = sqrt(2.0f)
	const double gamma_s = _gamma / 1.0f;
	const double gamma_d = _gamma / std::sqrt(2.0f);
	_l.create(_image.size(), CV_64FC1); // ������ͼ���С��ȵģ��洢�÷����Ȩ��
	_ul.create(_image.size(), CV_64FC1);
	_u.create(_image.size(), CV_64FC1);
	_ur.create(_image.size(), CV_64FC1);
	for (int y = 0; y < _image.rows; y++) {
		for (int x = 0; x < _image.cols; x++) {
			Vec3d color = (Vec3d)_image.at<Vec3b>(y, x);
			if (x - 1 >= 0) {
				Vec3d tmp = color - (Vec3d)_image.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = gamma_s * exp(-_beta * tmp.dot(tmp));
			}
			else _l.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d tmp = color - (Vec3d)_image.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gamma_d * exp(-_beta * tmp.dot(tmp));
			}
			else _ul.at<double>(y, x) = 0;
			if (y - 1 >= 0) {
				Vec3d tmp = color - (Vec3d)_image.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma * exp(-_beta * tmp.dot(tmp));
			}
			else _u.at<double>(y, x) = 0;
			if (x + 1 < _image.cols && y - 1 >= 0) {
				Vec3d tmp = color - (Vec3d)_image.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gamma_d * exp(-_beta * tmp.dot(tmp));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}
// ��ʼ������ͼ�Σ�ʹ�þ���
static void initmask_rect(Mat& _mask, Size _image_size, Rect _rect) {
	_mask.create(_image_size, CV_8UC1);
	_mask.setTo(GC_BGD);
	_rect.x = _rect.x > 0 ? _rect.x : 0;
	_rect.y = _rect.y > 0 ? _rect.y : 0;
	_rect.width = _rect.x + _rect.width > _image_size.width ? _image_size.width - _rect.x : _rect.width;
	_rect.height = _rect.y + _rect.height > _image_size.height ? _image_size.height - _rect.y : _rect.height;
	(_mask(_rect)).setTo(Scalar(GC_PR_FGD));
}
// ʹ��kmeans��ʼ����˹���ģ��
static void init_GMM(const Mat& _image, const Mat& _mask, GMM& _bgd_GMM, GMM& _fgd_GMM) {
	const int kmeans_iter = 10;
	Mat bgdLabels, fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;
	Point p;
	for (p.y = 0; p.y < _image.rows; p.y++) {
		for (p.x = 0; p.x < _image.cols; p.x++) {
			if (_mask.at<uchar>(p) == GC_BGD || _mask.at<uchar>(p) == GC_PR_BGD)
				bgdSamples.push_back((Vec3f)_image.at<Vec3b>(p));
			else
				fgdSamples.push_back((Vec3f)_image.at<Vec3b>(p));
		}
	}
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeans_iter, 0.0), 0, KMEANS_PP_CENTERS);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeans_iter, 0.0), 0, KMEANS_PP_CENTERS);
	//5.Learn GMM(���ݾ������������ÿ��GMM����еľ�ֵ��Э����Ȳ�����
	_bgd_GMM.initial();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		_bgd_GMM.add_point(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	_bgd_GMM.learning();

	_fgd_GMM.initial();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		_fgd_GMM.add_point(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	_fgd_GMM.learning();
}
// step1
static void assign_GMM(const Mat& _image, const Mat& _mask, const GMM& _bgd_GMM, const GMM& _fgd_GMM, Mat& _kn) {
	Point p;
	for (p.y = 0; p.y < _image.rows; p.y++) {
		for (p.x = 0; p.x < _image.cols; p.x++) {
			Vec3d color = (Vec3d)_image.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == GC_BGD || t == GC_PR_BGD)_kn.at<int>(p) = _bgd_GMM.choice(color);
			else _kn.at<int>(p) = _fgd_GMM.choice(color);
		}
	}
}
// step 2
static void learn_GMM_pa(const Mat& _image, const Mat& _mask, GMM& _bgd_GMM, GMM& _fgd_GMM, const Mat& _kn) {
	_bgd_GMM.initial();
	_fgd_GMM.initial();
	Point p;
	for (int i = 0; i < GMM::K; i++) {
		for (p.y = 0; p.y < _image.rows; p.y++) {
			for (p.x = 0; p.x < _image.cols; p.x++) {
				int tmp = _kn.at<int>(p);
				if (tmp == i) {
					if (_mask.at<uchar>(p) == GC_BGD || _mask.at<uchar>(p) == GC_PR_BGD)
						_bgd_GMM.add_point(tmp, _image.at<Vec3b>(p));
					else
						_fgd_GMM.add_point(tmp, _image.at<Vec3b>(p));
				}
			}
		}
	}
	_bgd_GMM.learning();
	_fgd_GMM.learning();
}
//����ͼ
static void getGraph(const Mat& _image, const Mat& _mask, const GMM& _bgd_GMM, const GMM& _fgd_GMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, GraphCut& _graph) {
	int vCount = _image.cols * _image.rows;
	int eCount = 2 * (4 * vCount - 3 * _image.cols - 3 * _image.rows + 2);
	_graph = GraphCut(vCount, eCount);
	Point p;
	for (p.y = 0; p.y < _image.rows; p.y++) {
		for (p.x = 0; p.x < _image.cols; p.x++) {
			int vnum = _graph.addVertex();
			Vec3b color = _image.at<Vec3b>(p);
			double f_source = 0, b_sink = 0;
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD) {
				f_source = -log(_bgd_GMM.data_weight(color));
				b_sink = -log(_fgd_GMM.data_weight(color));
			}
			else if (_mask.at<uchar>(p) == GC_BGD) b_sink = _lambda;
			else f_source = _lambda;
			_graph.addVertexWeights(vnum, f_source, b_sink);
			if (p.x > 0) {
				double w = _l.at<double>(p);
				_graph.addEdges(vnum, vnum - 1, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = _ul.at<double>(p);
				_graph.addEdges(vnum, vnum - _image.cols - 1, w);
			}
			if (p.y > 0) {
				double w = _u.at<double>(p);
				_graph.addEdges(vnum, vnum - _image.cols, w);
			}
			if (p.x < _image.cols - 1 && p.y > 0) {
				double w = _ur.at<double>(p);
				_graph.addEdges(vnum, vnum - _image.cols + 1, w);
			}
		}
	}
}
//���зָ� step3
static void estimateSegmentation(GraphCut& _graph, Mat& _mask) {
	_graph.maxFlow();
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD) {
				if (_graph.isSourceSegment(p.y * _mask.cols + p.x))
					_mask.at<uchar>(p) = GC_PR_FGD;
				else _mask.at<uchar>(p) = GC_PR_BGD;
			}
		}
	}
}
GrabCut2D::~GrabCut2D(void) {}
//GrabCut ������
void GrabCut2D::GrabCut(InputArray _image, InputOutputArray _mask, Rect rect, InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode) {
	std::cout << "Execute GrabCut Function: Please finish the code here!" << std::endl;
	//һ.�������ͣ�
	//���룺
	//cv::InputArray _image,     :�����colorͼ��(����-cv:Mat)
	//cv::Rect rect            :��ͼ���ϻ��ľ��ο�����-cv:Rect) 
	//int iterCount :           :ÿ�ηָ�ĵ�������������-int)
	//�м����
	//cv::InputOutputArray _bgdModel ��   ����ģ�ͣ��Ƽ�GMM)������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
	//cv::InputOutputArray _fgdModel :    ǰ��ģ�ͣ��Ƽ�GMM) ������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
	//���:
	//cv::InputOutputArray _mask  : ����ķָ��� (���ͣ� cv::Mat)
	//��. α�������̣�
	//1.Load Input Image: ����������ɫͼ��;
	Mat img = _image.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	//2.Init Mask: �þ��ο��ʼ��Mask��Labelֵ��ȷ��������0�� ȷ��ǰ����1�����ܱ�����2������ǰ����3��,���ο���������Ϊȷ�����������ο���������Ϊ����ǰ��;
	if (mode == GC_WITH_RECT) {
		initmask_rect(mask, img.size(), rect);
	}
	//3.Init GMM: ���岢��ʼ��GMM(����ģ����ɷָ�Ҳ�ɵõ�����������GMM��ɻ�ӷ֣�
	GMM bgdGMM(bgdModel);
	GMM fgdGMM(fgdModel);
	//4.Sample Points:ǰ������ɫ���������о��ࣨ������kmeans���������෽��Ҳ��)
	if (mode == GC_WITH_RECT || mode == GC_WITH_MASK) {
		init_GMM(img, mask, bgdGMM, fgdGMM);
	}
	if (iterCount <= 0) {
		return;
	}
	//6.Construct Graph������t-weight(�������n-weight��ƽ�����
	//����ƽ���������ļ�����GMMģ����ʵ��
	const double gamma = 50;
	const double beta = cal_Beta(img);
	Mat leftW, upleftW, upW, uprightW;
	cal_weight(img, leftW, upleftW, upW, uprightW, beta, gamma);
	//7.Estimate Segmentation(����maxFlow����зָ�)
	Mat compIdxs(img.size(), CV_32SC1);
	const double lambda = 9 * gamma;
	//���е���
	for (int i = 0; i < iterCount; i++) {
		GraphCut graph;
		assign_GMM(img, mask, bgdGMM, fgdGMM, compIdxs);
		learn_GMM_pa(img, mask, bgdGMM, fgdGMM, compIdxs);
		getGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		estimateSegmentation(graph, mask);
	}
	//8.Save Result�������������mask�������mask��ǰ�������Ӧ�Ĳ�ɫͼ�񱣴����ʾ�ڽ��������У�
}