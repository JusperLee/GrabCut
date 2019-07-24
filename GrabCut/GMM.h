#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class GMM {
public:	
	static const int K = 5;//���������е�ʵ�֣�Ϊ5
	
	GMM(cv::Mat& _model);//�� model �ж�ȡ�������洢
	
	double color_k_prob(int, const cv::Vec3d) const;//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
	double data_weight(const cv::Vec3d) const;//������Ȩ��
	int choice(const cv::Vec3d) const;//һ����ɫӦ���������ĸ����
	void initial();//��ʼ��
	void add_point(int, const cv::Vec3d);//��ӵ����ĵ�
	void learning();
private:
	void cal_covariances(int);//����Э���������������ʽ��ֵ
	cv::Mat model;//�洢GMMģ��
	double* pi, * mean, * cov;//Ȩ�ء���ֵ��Э����
	double cov_t[K][3][3];//Э�������
	double cov_det[K];//Э���������ʽֵ
	
	double rgb_sums[K][3];
	double rgb_covs[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif
