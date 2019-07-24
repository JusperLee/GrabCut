#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "GrabCut.h"
#include <iostream>
using namespace std;
using namespace cv;

const Scalar BLUE = Scalar(255, 0, 0); // 背景 
const Scalar GREEN = Scalar(0, 255, 0);// 前景
const Scalar LIGHTBLUE = Scalar(255, 255, 160);// 可能的背景
const Scalar PINK = Scalar(230, 130, 255); // 可能的前景
const Scalar RED = Scalar(0, 0, 255);// 矩形颜色

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;// 按“CTRL”键时，flags的值返回
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;// 按“SHIFT”键时，flags的值返回


// 将comMask的值复制到binMask
static void getBinMask(const Mat& comMask, Mat& binMask)
{
	if (comMask.empty() || comMask.type() != CV_8UC1)
		CV_Error(CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
		binMask.create(comMask.size(), CV_8UC1);
	binMask = comMask & 1;
}


class GCApplication
{
public:
	enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
	static const int radius = 2; // 圆的半径
	static const int thickness = -1; // 圆线的粗细 （画线作用）

	void reset();
	void setImageAndWinName(const Mat& _image, const string& _winName);
	void borderMatting();
	void showImage();
	void mouseClick(int event, int x, int y, int flags, void* param);
	int nextIter();
	int getIterCount() const { return iterCount; }
	void borderMatting(const cv::Mat&, cv::Mat&);
private:
	void setRectInMask();
	void setLblsInMask(int flags, Point p, bool isPr);

	const string* winName;
	const Mat* image;
	Mat mask, alphaMask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
	GrabCut2D gc;
};


