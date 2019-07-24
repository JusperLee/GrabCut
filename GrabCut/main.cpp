#include <iostream>
#include <opencv2/opencv.hpp>
#include "GCApplication.h"
static void help()
{
	std::cout << "\n此程序演示了GrabCut分割 - 选择区域中的对象\n"
		"然后抓取将尝试将其分割出来。\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\n选择要分割的对象周围的矩形区域\n" <<
		"\n快捷键: \n"
		"\tESC - 退出程序\n"
		"\tr - 恢复原始图像\n"
		"\tn - 下一次迭代\n"
		"\n"
		"\t鼠标左键 - 设置矩形\n"
		"\n"
		"\tCTRL+鼠标左键 - 选择背景像素\n"
		"\tSHIFT+鼠标左键 - 选择前景像素\n"
		"\n"
		"\tCTRL+鼠标右键 - 选择可能是背景像素\n"
		"\tSHIFT+鼠标右键 - 选择可能是前景像素\n" << endl;
}


GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
	gcapp.mouseClick( event, x, y, flags, param );
}


int main()
{
	//cout << "请输入文件路径+文件名（例：F:\\Data\\Ls_Data\\00\\0050.bmp）\n";
	string pic_name = "C:\\Users\\de'l'l\\Desktop\\1.jpg";
	// cin >> pic_name;
	string filename = pic_name;
	Mat image = imread( filename, 1 );
	/*
	Size s;
	s.height = image.rows / 2;
	s.width = image.rows / 2;
	resize(image, image, s);
	
	*/

	if (pic_name.empty())
	{
		cout << "\n , 空文件名" << endl;
		return 1;
	}
	if( image.empty() )
	{
		cout << "\n , 无法读取图像文件名 " << filename << endl;
		return 1;
	}

	help();

	const string winName = "image";
	namedWindow( winName, WINDOW_AUTOSIZE );
	setMouseCallback( winName.c_str(), on_mouse, 0 );

	gcapp.setImageAndWinName( image, winName );
	gcapp.showImage();

	for(;;)
	{
		char c = (char) waitKey(0);
		switch( c )
		{
		case '\x1b':
			cout << "推出 ..." << endl;
			goto exit_main;
		case 'r':
			cout <<"r"<< endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'n':
			int iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if( newIterCount > iterCount )
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "必须确定矩形>" << endl;
			break;
		}
	}

exit_main:
	destroyWindow( winName );
	return 0;
}