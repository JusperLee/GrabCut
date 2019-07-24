#include <iostream>
#include <opencv2/opencv.hpp>
#include "GCApplication.h"
static void help()
{
	std::cout << "\n�˳�����ʾ��GrabCut�ָ� - ѡ�������еĶ���\n"
		"Ȼ��ץȡ�����Խ���ָ������\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nѡ��Ҫ�ָ�Ķ�����Χ�ľ�������\n" <<
		"\n��ݼ�: \n"
		"\tESC - �˳�����\n"
		"\tr - �ָ�ԭʼͼ��\n"
		"\tn - ��һ�ε���\n"
		"\n"
		"\t������ - ���þ���\n"
		"\n"
		"\tCTRL+������ - ѡ�񱳾�����\n"
		"\tSHIFT+������ - ѡ��ǰ������\n"
		"\n"
		"\tCTRL+����Ҽ� - ѡ������Ǳ�������\n"
		"\tSHIFT+����Ҽ� - ѡ�������ǰ������\n" << endl;
}


GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
	gcapp.mouseClick( event, x, y, flags, param );
}


int main()
{
	//cout << "�������ļ�·��+�ļ���������F:\\Data\\Ls_Data\\00\\0050.bmp��\n";
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
		cout << "\n , ���ļ���" << endl;
		return 1;
	}
	if( image.empty() )
	{
		cout << "\n , �޷���ȡͼ���ļ��� " << filename << endl;
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
			cout << "�Ƴ� ..." << endl;
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
				cout << "����ȷ������>" << endl;
			break;
		}
	}

exit_main:
	destroyWindow( winName );
	return 0;
}