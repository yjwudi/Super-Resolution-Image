#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "my_image.h"

#define alpha 1.25
#define RATIO 3

using namespace cv;
using namespace std;

FILE *fp=fopen("output.txt", "w+");

Mat sr_single(Mat ori_img);
void print_img_data(my_image img_data);
Mat data_to_img(Mat high_img, my_image img_data);
Mat back_project(Mat high_img_data, Mat ori_image);
int main(int argc, char* argv[])
{
    system("ulimit -s unlimited");
    Mat img = imread(argv[1]);

    cout << "start: " << endl;
    Mat sr_img = sr_single(img);
    cout << "end\n";

    fclose(fp);
    return 0;
}
Mat sr_single(Mat ori_img)
{
    int level = 0, total_level = 3; //(int)(log(alpha)/log(RATIO*1.0)+0.5);
   // int patch_size[2] = {5, 5};
   // my_image recons_img = l_img;
    Mat up_img, down_img, tmp_high_img, tmp_low_img, high_img;
    my_image high_img_data, low_img_data, ori_img_data;
    int up_row = int(ori_img.rows*alpha+0.5), up_col = int(ori_img.cols*alpha+0.5);
    int down_row = ori_img.rows/(alpha*1.0), down_col = ori_img.cols/(alpha*1.0);
    int times = 0;
    double ratio = 1.0;
    ori_img_data = my_image(ori_img);
    tmp_high_img = ori_img, tmp_low_img = ori_img;
    char filename[10] = {"sri.png"};
    //img_pair sr_data = img_pair(l_img, 1);
    for(level = 0; level < total_level; level++)
    {
        printf("level: %d\n", level);
        printf("up_row, up_col, down_row, down_col: %d %d %d %d\n", up_row, up_col, down_row, down_col);
        resize(ori_img, up_img, Size(up_row, up_col), (0,0), (0,0));
        resize(ori_img, down_img, Size(down_row, down_col), (0,0), (0,0));
        
        ratio *= alpha;
        high_img_data = my_image(up_img), low_img_data = my_image(down_img);
	high_img_data.construct(ori_img_data, low_img_data, ratio);
	//print_img_data(high_img_data);
        high_img = data_to_img(up_img, high_img_data);
	filename[2] = level+'0';
        imwrite(filename, high_img);
	//high_img = back_project(high_img, ori_img);
        //tmp_high_img = high_img, tmp_low_img = down_img;
        up_row = int(up_row*alpha+0.5), up_col = int(up_col*alpha+0.5);
        down_row = down_row/(alpha*1.0), down_col = down_col/(alpha*1.0);
    }
    return high_img;
}
void print_img_data(my_image img_data)
{
    int i = 0, j = 0, row = img_data.img_row, col = img_data.img_col;
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
            {
		fprintf(fp, "%d ", img_data.r_data[i][j]);
		fprintf(fp, "%d ", img_data.g_data[i][j]);
		fprintf(fp, "%d ", img_data.b_data[i][j]);
	    }
        fprintf(fp, "\n");
    }
}
Mat data_to_img(Mat high_img, my_image img_data)
{
    int i = 0, j = 0;
    for(i = 0; i < img_data.img_row; i++)
    {
        for(j = 0; j < img_data.img_col; j++)
        {
            high_img.at<Vec3b>(i,j)[0] = img_data.b_data[i][j];
            high_img.at<Vec3b>(i,j)[1] = img_data.g_data[i][j];
            high_img.at<Vec3b>(i,j)[2] = img_data.r_data[i][j];
        }
    }
    return high_img;
}
Mat back_project(Mat high_img, Mat ori_img)
{
    int i = 0, j = 0, diff;
    Mat back_img, diff_img, high_diff_img;
    my_image back_img_data, ori_img_data, diff_img_data;
    diff_img = ori_img;
    resize(high_img, back_img, Size(ori_img.rows, ori_img.cols), (0,0), (0,0));
    back_img_data = my_image(back_img), ori_img_data = my_image(ori_img), diff_img_data = my_image(ori_img);
    for(i = 0; i < ori_img.rows; i++)
        for(j = 0; j < ori_img.cols; j++)
        {
            diff_img_data.b_data[i][j] = ori_img_data.b_data[i][j] - back_img_data.b_data[i][j];
            diff_img_data.r_data[i][j] = ori_img_data.r_data[i][j] - back_img_data.r_data[i][j];
            diff_img_data.g_data[i][j] = ori_img_data.g_data[i][j] - back_img_data.g_data[i][j];
        }
    diff_img = data_to_img(diff_img, diff_img_data);
    resize(diff_img, high_diff_img,  Size(high_img.rows, high_img.cols), (0,0), (0,0));
    my_image high_img_data = my_image(high_img), high_diff_img_data = my_image(high_diff_img);
    for(i = 0; i < high_img.rows; i++)
        for(j = 0; j < high_img.cols; j++)
        {
            high_img_data.b_data[i][j] += high_diff_img_data.b_data[i][j];
            high_img_data.g_data[i][j] += high_diff_img_data.g_data[i][j];
            high_img_data.r_data[i][j] += high_diff_img_data.r_data[i][j];
        }
    return data_to_img(high_img, high_img_data);

}


