#include "GMM.h"
#include <vector>
using namespace std;
using namespace cv;



//GMM的构造函数，从 model 中读取参数并存储
GMM::GMM(Mat& _model) {
	if (_model.empty()) {
		//每一个高斯分量的权重π、每个高斯分量的均值向量u（因为有RGB三个通道，故为三个元素向量）
		//和协方差矩阵∑（因为有RGB三个通道，故为3x3矩阵）
		_model.create(1, 13 * K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	// 先排完componentsCount个coefs，再3* componentsCount个mean, 再3*3*componentsCount个cov
	pi = model.ptr<double>(0);
	mean = pi + K;
	cov = mean + 3 * K;
	
	for (int i = 0; i < K; i++)
		if (pi[i] > 0)//如果某个项的权重不为0，则计算其协方差的逆和行列式
			cal_covariances(i);// 计算每个像素属于该高斯模型的概率（也就是数据能量项)
}
double GMM::color_k_prob(int _i, const Vec3d _color) const {
	double res = 0;
	if (pi[_i] > 0) {
		Vec3d diff = _color;
		double* m = mean + 3 * _i; // 得到第i个类别的三个mean
		diff[0] -= m[0]; // 计算（x-μ）
		diff[1] -= m[1]; 
		diff[2] -= m[2];
		double mult = diff[0] * (diff[0] * cov_t[_i][0][0] + diff[1] * cov_t[_i][1][0] + diff[2] * cov_t[_i][2][0])
			+ diff[1] * (diff[0] * cov_t[_i][0][1] + diff[1] * cov_t[_i][1][1] + diff[2] * cov_t[_i][2][1])
			+ diff[2] * (diff[0] * cov_t[_i][0][2] + diff[1] * cov_t[_i][1][2] + diff[2] * cov_t[_i][2][2]);
		// 混合高斯密度模型
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
// 像素集中所有像素的RGB三个通道的和sums（用来计算均值），还有它的prods（用来计算协方差）
void GMM::initial() {
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++)
			rgb_sums[i][j] = 0;
		for (int p = 0; p < 3; p++) {
			for (int q = 0; q < 3; q++) {
				rgb_covs[i][p][q] = 0;
			}
		}
		sampleCounts[i] = 0; // 该类别的像素个数
	}
	totalSampleCount = 0; // 总像素个数 （samplecounts/totalsamplecounts=高斯模型权重）
}
//添加单个的点
void GMM::add_point(int _i, const Vec3d _color) {

	for (int i = 0; i < 3; i++) {
		rgb_sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			rgb_covs[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}
//根据添加的数据，计算新的参数结果
void GMM::learning() {
	const double variance = 0.01;
	for (int i = 0; i < K; i++) {
		int n = sampleCounts[i];
		if (n == 0)	pi[i] = 0;
		else {
			//计算高斯模型新的参数
			//权重
			pi[i] = 1.0 * n / totalSampleCount;
			//均值
			double* m = mean + 3 * i;
			for (int j = 0; j < 3; j++) {
				m[j] = rgb_sums[i][j] / n;
			}
			//协方差
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					c[p * 3 + q] = rgb_covs[i][p][q] / n - m[p] * m[q];
				}
			}
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			//相当于如果行列式小于等于0，（对角线元素）增加白噪声，避免其变
            //为退化（降秩）协方差矩阵（不存在逆矩阵，但后面的计算需要计算逆矩阵）
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
		//行列式的值
		double dtrm = cov_det[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//计算逆矩阵
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



