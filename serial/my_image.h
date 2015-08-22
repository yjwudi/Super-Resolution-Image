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
		void construct(my_image ori_img_data, my_image low_img_data, double ratio);
		void query(double** &sample_data, double **&train_data, int sample_row, int sample_col, int train_row, double** &aim_high_res);
		void find_nearest(int row, int col, my_image ori_img_data, my_image low_img_data, int &x, int &y);

};
my_image::my_image()
{
}
my_image::my_image(Mat img)
{
	int i = 0, j = 0, row = 0, col = 0;
	int R = 0, G = 0, B = 0, y = 0, cb = 0, cr = 0;
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
void my_image::construct(my_image ori_img_data, my_image low_img_data, double ratio)
{
	int i = 0, j = 0, s = 0, t = 0;
	int row = ori_img_data.img_row, col = ori_img_data.img_col;
	int aim_x, aim_y;
	for(i = 0; i < row; i+=2)
		for(j = 0; j < col; j+=2)
		{ 
			find_nearest(i, j, ori_img_data, low_img_data, aim_x, aim_y);
			aim_x = aim_x*ratio, aim_y = aim_y*ratio;
			int start_row = i*ratio, start_col = j*ratio;
			s = (int)(2*ratio+0.5)-1, t = (int)(2*ratio+0.5)-1;

			if(start_row+s<img_row&&start_col+t<img_col&&aim_x+s<ori_img_data.img_row&&aim_y+t<ori_img_data.img_col)
			{
				for(s = 0; s < (int)(2*ratio+0.5); s++)
					for(t = 0; t < (int)(2*ratio+0.5); t++)
					{
						r_data[start_row+s][start_col+t] = ori_img_data.r_data[aim_x+s][aim_y+t];
						g_data[start_row+s][start_col+t] = ori_img_data.g_data[aim_x+s][aim_y+t];
						b_data[start_row+s][start_col+t] = ori_img_data.b_data[aim_x+s][aim_y+t];

					}
			}
		}
}
void my_image::find_nearest(int row, int col, my_image ori_img_data, my_image low_img_data, int &x, int &y)
{
	int i = 0, j = 0, s = 0, t = 0, diff, min_diff=1000000, pos_x, pos_y;
	for(s = 0; s < low_img_data.img_row-2; s++)
	{
		for(t = 0; t < low_img_data.img_col-2; t++)
		{
			if(row+1 < ori_img_data.img_row && col+1 < ori_img_data.img_col)
			{
				diff = 0;
				for(i = 0; i < 2; i++)
					for(j = 0; j < 2; j++)
					{
						diff += abs(ori_img_data.r_data[row+i][col+j]-low_img_data.r_data[s+i][t+j]) + abs(ori_img_data.g_data[row+i][col+j]-low_img_data.g_data[s+i][t+j]) + abs(ori_img_data.b_data[row+i][col+j]-low_img_data.b_data[s+i][t+j]);
					}
				if(diff < min_diff)
				{
					min_diff = diff;
					pos_x = s, pos_y = t;
				}
			}
		}
	}
	x = pos_x, y = pos_y;
}


#endif // MY_IMAGE_H_INCLUDED
