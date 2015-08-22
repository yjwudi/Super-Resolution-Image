#ifndef MY_IMAGE_H_INCLUDED
#define MY_IMAGE_H_INCLUDED

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mpi.h>


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
	int **tmp_r_data, **tmp_g_data, **tmp_b_data;
    //MatrixXi y_data, cb_data, cr_data;
	int img_row, img_col; //y_data的行列
	int *local_rgb, *total_rgb;
	Mat _img;

	my_image();
	my_image(Mat img);
	void construct(my_image ori_img_data, my_image low_img_data, double ratio);
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
	init2D(tmp_r_data, row, col), init2D(tmp_g_data, row, col), init2D(tmp_b_data, row, col);
	total_rgb = new int[img_row*img_col*3+5], local_rgb = new int[img_row*img_col*3+5];
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
	int my_rank, comm_sz;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	int row_num = ori_img_data.img_row/comm_sz, col_num = ori_img_data.img_col;
	int start_row = my_rank*row_num, end_row = start_row+row_num;
	int l_x = 0, l_y = 0, aim_x = 0, aim_y = 0, high_x = 0, high_y = 0;

	int high_row_num = img_row/comm_sz, high_col_num = img_col;
	int high_start_row = my_rank*high_row_num, local_len = 0;

	if(my_rank == comm_sz-1)
		end_row = ori_img_data.img_row;
	for(i = start_row; i < end_row; i+=2)
		for(j = 0; j < col_num; j+=2)
		{
			//l_x = i*2, l_y = j*2;
			find_nearest(i, j, ori_img_data, low_img_data, aim_x, aim_y);
			aim_x = aim_x*ratio, aim_y = aim_y*ratio;
			high_x = i*ratio, high_y = j*ratio;
			s = (int)(2*ratio+0.5)-1, t = (int)(2*ratio+0.5)-1;
			if(high_x+s<img_row&&high_y+t<img_col&&aim_x+s<ori_img_data.img_row&&aim_y+t<ori_img_data.img_col)
			{
			for(s = 0; s < (int)(2*ratio+0.5); s++)
				for(t = 0; t < (int)(2*ratio+0.5); t++)
				{
					tmp_r_data[high_x+s][high_y+t] = ori_img_data.r_data[aim_x+s][aim_y+t];
					tmp_g_data[high_x+s][high_y+t] = ori_img_data.g_data[aim_x+s][aim_y+t];
					tmp_b_data[high_x+s][high_y+t] = ori_img_data.b_data[aim_x+s][aim_y+t];
				}
			}
		}
	for(i = high_start_row; i < high_start_row+high_row_num; i++)
		for(j = 0; j < high_col_num; j++)
		{
			local_rgb[local_len++] = tmp_r_data[i][j];
			local_rgb[local_len++] = tmp_g_data[i][j];
			local_rgb[local_len++] = tmp_b_data[i][j];
		}
	if(my_rank != 0)
	{
		MPI_Send(local_rgb, local_len, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		for(i = 0; i < local_len; i++)
			total_rgb[i] = local_rgb[i];
		for(int source = 1; source < comm_sz; source++)
		{
		    MPI_Recv(local_rgb, local_len, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    for(i = local_len*source, s = 0; i < local_len*(source+1); i++, s++)
		    	total_rgb[i] = local_rgb[s];
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
