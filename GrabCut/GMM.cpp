#include "GMM.h"
#include <vector>
using namespace std;
using namespace cv;



//GMM�Ĺ��캯������ model �ж�ȡ�������洢
GMM::GMM(Mat& _model) {
	if (_model.empty()) {
		//ÿһ����˹������Ȩ�ئС�ÿ����˹�����ľ�ֵ����u����Ϊ��RGB����ͨ������Ϊ����Ԫ��������
		//��Э�������ƣ���Ϊ��RGB����ͨ������Ϊ3x3����
		_model.create(1, 13 * K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	// ������componentsCount��coefs����3* componentsCount��mean, ��3*3*componentsCount��cov
	pi = model.ptr<double>(0);
	mean = pi + K;
	cov = mean + 3 * K;
	
	for (int i = 0; i < K; i++)
		if (pi[i] > 0)//���ĳ�����Ȩ�ز�Ϊ0���������Э������������ʽ
			cal_covariances(i);// ����ÿ���������ڸø�˹ģ�͵ĸ��ʣ�Ҳ��������������)
}
double GMM::color_k_prob(int _i, const Vec3d _color) const {
	double res = 0;
	if (pi[_i] > 0) {
		Vec3d diff = _color;
		double* m = mean + 3 * _i; // �õ���i����������mean
		diff[0] -= m[0]; // ���㣨x-�̣�
		diff[1] -= m[1]; 
		diff[2] -= m[2];
		double mult = diff[0] * (diff[0] * cov_t[_i][0][0] + diff[1] * cov_t[_i][1][0] + diff[2] * cov_t[_i][2][0])
			+ diff[1] * (diff[0] * cov_t[_i][0][1] + diff[1] * cov_t[_i][1][1] + diff[2] * cov_t[_i][2][1])
			+ diff[2] * (diff[0] * cov_t[_i][0][2] + diff[1] * cov_t[_i][1][2] + diff[2] * cov_t[_i][2][2]);
		// ��ϸ�˹�ܶ�ģ��
		res = 1.0f / sqrt(cov_det[_i]) * exp(-0.5f * mult);
	}
	return res;
}

double GMM::data_weight(const Vec3d _color)const {
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += pi[ci] * color_k_prob(ci, _color);
	return res;
}

int GMM::choice(const Vec3d _color) const {
	int k = 0;
	double max = 0;
	for (int i = 0; i < K; i++) {
		double p = color_k_prob(i, _color);
		if (p > max) {
			k = i;
			max = p;
		}
	}
	return k;
}
// ���ؼ����������ص�RGB����ͨ���ĺ�sums�����������ֵ������������prods����������Э���
void GMM::initial() {
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++)
			rgb_sums[i][j] = 0;
		for (int p = 0; p < 3; p++) {
			for (int q = 0; q < 3; q++) {
				rgb_covs[i][p][q] = 0;
			}
		}
		sampleCounts[i] = 0; // ���������ظ���
	}
	totalSampleCount = 0; // �����ظ��� ��samplecounts/totalsamplecounts=��˹ģ��Ȩ�أ�
}
//��ӵ����ĵ�
void GMM::add_point(int _i, const Vec3d _color) {

	for (int i = 0; i < 3; i++) {
		rgb_sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			rgb_covs[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}
//������ӵ����ݣ������µĲ������
void GMM::learning() {
	const double variance = 0.01;
	for (int i = 0; i < K; i++) {
		int n = sampleCounts[i];
		if (n == 0)	pi[i] = 0;
		else {
			//�����˹ģ���µĲ���
			//Ȩ��
			pi[i] = 1.0 * n / totalSampleCount;
			//��ֵ
			double* m = mean + 3 * i;
			for (int j = 0; j < 3; j++) {
				m[j] = rgb_sums[i][j] / n;
			}
			//Э����
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					c[p * 3 + q] = rgb_covs[i][p][q] / n - m[p] * m[q];
				}
			}
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			//�൱���������ʽС�ڵ���0�����Խ���Ԫ�أ����Ӱ��������������
            //Ϊ�˻������ȣ�Э������󣨲���������󣬵�����ļ�����Ҫ���������
			if (dtrm <= std::numeric_limits<double>::epsilon()) {
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}

			cal_covariances(i);
		}
	}
}

void GMM::cal_covariances(int _i) {
	if (pi[_i] > 0) {
		double* c = cov + 9 * _i;
		//����ʽ��ֵ
		double dtrm = cov_det[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//���������
		cov_t[_i][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		cov_t[_i][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		cov_t[_i][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		cov_t[_i][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		cov_t[_i][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		cov_t[_i][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		cov_t[_i][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		cov_t[_i][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		cov_t[_i][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}



