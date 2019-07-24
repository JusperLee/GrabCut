#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class GMM {
public:	
	static const int K = 5;//按照论文中的实现，为5
	
	GMM(cv::Mat& _model);//从 model 中读取参数并存储
	
	double color_k_prob(int, const cv::Vec3d) const;//计算某个颜色属于某个组件的可能性（高斯概率）
	double data_weight(const cv::Vec3d) const;//数据项权重
	int choice(const cv::Vec3d) const;//一个颜色应该是属于哪个组件
	void initial();//初始化
	void add_point(int, const cv::Vec3d);//添加单个的点
	void learning();
private:
	void cal_covariances(int);//计算协方差矩阵的逆和行列式的值
	cv::Mat model;//存储GMM模型
	double* pi, * mean, * cov;//权重、均值、协方差
	double cov_t[K][3][3];//协方差的逆
	double cov_det[K];//协方差的行列式值
	
	double rgb_sums[K][3];
	double rgb_covs[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif
