#ifndef MY_IMAGE_H_INCLUDED
#define MY_IMAGE_H_INCLUDED

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <cstdlib>
#include <cstring>
#include <cmath>


using namespace cv;
using namespace std;

template<class T>
void init2D(T** &_array, int row, int col);
template<class T>
void delete2D(T** &_array, int row, int col);

class my_image
{
    public:
	int** r_data, **g_data, **b_data;
    //MatrixXi y_data, cb_data, cr_data;
	int img_row, img_col; //y_data的行列
	Mat _img;

	my_image();
	my_image(Mat img);

};
my_image::my_image()
{
}
my_image::my_image(Mat img)
{
	int i = 0, j = 0, row = 0, col = 0;
	int R = 0, G = 0, B = 0;
	img_row = img.rows, img_col = img.cols, _img = img;
	row = img_row, col = img_col;
	init2D(r_data, row, col), init2D(g_data, row, col), init2D(b_data, row, col);
	for(i = 0; i < row; i++)
	{
		for(j = 0; j < col; j++)
		{
			B = (int)img.at<Vec3b>(i,j)[0];
			G = (int)img.at<Vec3b>(i,j)[1];
			R = (int)img.at<Vec3b>(i,j)[2];
	        r_data[i][j] = R, g_data[i][j] = G, b_data[i][j] = B;
		}
	}
}
template<class T>
void init2D(T** &_array, int row, int col)
{
	int i = 0;
	_array = new T*[row];
	for(i = 0; i < row; i++)
		_array[i] = new T[col];
}
template<class T>
void delete2D(T** &_array, int row, int col)
{
	int i = 0;
	for(i = 0; i < row; i++)
          delete []_array[i];
     delete []_array;
}


#endif // MY_IMAGE_H_INCLUDED
