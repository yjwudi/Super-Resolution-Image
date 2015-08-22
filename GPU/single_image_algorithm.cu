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

__global__ void find_nearest(int *r_data, int *g_data, int *b_data, size_t o_pitch, int ori_col, int *train_r, int *train_g, int *train_b, size_t d_pitch, int train_data_row, int train_data_col, int *dx, int *dy)
{
    int thread_id = threadIdx.x;
    int j = 0, s = 0, t = 0, k = 0;
    int *f_r_row, *f_g_row, *f_b_row, *s_r_row, *s_g_row, *s_b_row; 
    int *tf_r_row, *tf_g_row, *tf_b_row, *ts_r_row, *ts_g_row, *ts_b_row;
    int diff = 0, min_diff = 10000;
    if(thread_id%2 == 0)
    {
    f_r_row = (int*)((char*)r_data);
    f_g_row = (int*)((char*)g_data);
    f_b_row = (int*)((char*)b_data);
    s_r_row = (int*)((char*)r_data + o_pitch);
    s_g_row = (int*)((char*)r_data + o_pitch);
    s_b_row = (int*)((char*)r_data + o_pitch);
    k = thread_id;
        for(s = 0; s < train_data_row-2; s++)
        {
            tf_r_row = (int*)((char*)train_r + s*d_pitch);
            tf_g_row = (int*)((char*)train_g + s*d_pitch);
            tf_b_row = (int*)((char*)train_b + s*d_pitch);
            ts_r_row = (int*)((char*)train_r + (s+1)*d_pitch);
            ts_g_row = (int*)((char*)train_g + (s+1)*d_pitch);
            ts_b_row = (int*)((char*)train_b + (s+1)*d_pitch);
            for(t = 0; t < train_data_col-2; t++)
            {
                if(k+1 < ori_col)
                {
                diff = 0;
                for(j = 0; j < 2; j++)
                {
                    diff = diff + abs(f_r_row[k+j]-tf_r_row[t+j]) + abs(f_g_row[k+j]-tf_g_row[t+j]) + abs(f_b_row[k+j]-tf_b_row[t+j]) + abs(s_r_row[k+j]-ts_r_row[t+j]) + abs(s_g_row[k+j]-ts_g_row[t+j]) + abs(s_b_row[k+j]-ts_b_row[t+j]);
                }
                if(diff < min_diff)
                {
                    min_diff = diff;
                    dx[k/2] = s, dy[k/2] = t;
                }
                }
            }
        }
    }

}

//FILE *fp=fopen("output.txt", "w+");

void sr_single(Mat ori_img);
//void print_img_data(my_image img_data);
Mat data_to_img(Mat high_img, my_image img_data);
void construct(my_image ori_img_data, my_image low_img_data, double ratio, my_image high_img_data);
int main(int argc, char* argv[])
{
    system("ulimit -s unlimited");
    Mat img = imread(argv[1]);

    cout << "start: " << endl;
    sr_single(img);
    cout << "end\n";

    //fclose(fp);
    return 0;
}
void sr_single(Mat ori_img)
{
    int level = 0, total_level = 3;
    Mat up_img, down_img, tmp_high_img, tmp_low_img, high_img;
    my_image high_img_data, low_img_data, ori_img_data;
    int up_row = int(ori_img.rows*alpha+0.5), up_col = int(ori_img.cols*alpha+0.5);
    int down_row = ori_img.rows/(alpha*1.0), down_col = ori_img.cols/(alpha*1.0);
    double ratio = 1.0;
    char filename[10] = {"sri.png"};

    ori_img_data = my_image(ori_img);
    tmp_high_img = ori_img, tmp_low_img = ori_img;

    for(level = 0; level < total_level; level++)
    {
        printf("level: %d\n", level);
        printf("up_row, up_col, down_row, down_col: %d %d %d %d\n", up_row, up_col, down_row, down_col);
        resize(ori_img, up_img, Size(up_row, up_col));
        resize(ori_img, down_img, Size(down_row, down_col));

        ratio *= alpha;
        high_img_data = my_image(up_img), low_img_data = my_image(down_img);
        construct(ori_img_data, low_img_data, ratio, high_img_data);
        high_img = data_to_img(up_img, high_img_data);
        filename[2] = level+'0';
        imwrite(filename, high_img);

        up_row = int(up_row*alpha+0.5), up_col = int(up_col*alpha+0.5);
        down_row = down_row/(alpha*1.0), down_col = down_col/(alpha*1.0);
    }

}

void construct(my_image ori_img_data, my_image low_img_data, double ratio, my_image high_img_data)
{
    size_t d_pitch, o_pitch;
    int i = 0, j = 0, s = 0, t = 0, k = 0;
    int train_data_row = low_img_data.img_row, train_data_col = low_img_data.img_col;
    int row = ori_img_data.img_row, col = ori_img_data.img_col;
printf("row, col: %d %d\n", row, col);
    int *x, *y, *dx, *dy;
    int ori_r[2][col], ori_g[2][col], ori_b[2][col];
    int train_r[train_data_row][train_data_col], train_g[train_data_row][train_data_col], train_b[train_data_row][train_data_col];
    int *d_train_r, *d_train_g, *d_train_b, *d_r, *d_g, *d_b;
    int start_row, start_col, aim_row, aim_col;

    //注意rgb顺序
    for(i = 0; i < train_data_row; i++)
        for(j = 0; j < train_data_col; j++)
        {
            train_r[i][j] = low_img_data.r_data[i][j];
            train_g[i][j] = low_img_data.g_data[i][j];
            train_b[i][j] = low_img_data.b_data[i][j];

        }
    //这几个d_pitch输出看一下
    cudaMallocPitch((void**)&d_train_r, &d_pitch, sizeof(int)*train_data_col, train_data_row);
    cudaMallocPitch((void**)&d_train_g, &d_pitch, sizeof(int)*train_data_col, train_data_row);
    cudaMallocPitch((void**)&d_train_b, &d_pitch, sizeof(int)*train_data_col, train_data_row);
    cudaMemcpy2D(d_train_r, d_pitch, train_r, sizeof(int)*train_data_col, sizeof(int)*train_data_col, train_data_row, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_train_g, d_pitch, train_g, sizeof(int)*train_data_col, sizeof(int)*train_data_col, train_data_row, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_train_b, d_pitch, train_b, sizeof(int)*train_data_col, sizeof(int)*train_data_col, train_data_row, cudaMemcpyHostToDevice);

    x = new int[col], y = new int[col];
    cudaMalloc((void **)&dx, sizeof(int)*col);
    cudaMalloc((void **)&dy, sizeof(int)*col);
    cudaMallocPitch((void**)&d_r, &o_pitch, sizeof(int)*col, 2);
    cudaMallocPitch((void**)&d_g, &o_pitch, sizeof(int)*col, 2);
    cudaMallocPitch((void**)&d_b, &o_pitch, sizeof(int)*col, 2);
    for(i = 0; i < row-1; i+=2)
    {
        //这一段通过直接cudamemcpy原数组不借助二维数组赋值可不可以
        for(j = 0; j < col; j++)
        {
            ori_r[0][j] = ori_img_data.r_data[i][j], ori_r[1][j] = ori_img_data.r_data[i+1][j];
            ori_g[0][j] = ori_img_data.g_data[i][j], ori_g[1][j] = ori_img_data.g_data[i+1][j];
            ori_b[0][j] = ori_img_data.b_data[i][j], ori_b[1][j] = ori_img_data.b_data[i+1][j];
        }
        cudaMemcpy2D(d_r, o_pitch, ori_r, sizeof(int)*col, sizeof(int)*col, 2, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_g, o_pitch, ori_g, sizeof(int)*col, sizeof(int)*col, 2, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_b, o_pitch, ori_b, sizeof(int)*col, sizeof(int)*col, 2, cudaMemcpyHostToDevice);

        //dim3 blocks(1,2);
        //dim3 threads(col/2,2);
        find_nearest<<<1, col>>>(d_r, d_g, d_b, o_pitch, col, d_train_r, d_train_g, d_train_b, d_pitch, train_data_row, train_data_col, dx, dy);
        cudaMemcpy(x, dx, sizeof(int)*(col/2), cudaMemcpyDeviceToHost);
        cudaMemcpy(y, dy, sizeof(int)*(col/2), cudaMemcpyDeviceToHost);

        for(k = 0; k < col/2; k++)
        {
            aim_row = x[k]*ratio, aim_col = y[k]*ratio;
            start_row = aim_row*ratio, start_col = aim_col*ratio;
            s = (int)(2*ratio+0.5), t = (int)(2*ratio+0.5);
            if(start_row+s<high_img_data.img_row&&start_col+t<high_img_data.img_col&&aim_row+s<ori_img_data.img_row&&aim_col+t<ori_img_data.img_col)
            {//printf("k: %d col: %d\n", k, col);
              for(s = 0; s < (int)(2*ratio+0.5); s++)
                for(t = 0; t < (int)(2*ratio+0.5); t++)
                {
                    high_img_data.r_data[start_row+s][start_col+t] = ori_img_data.r_data[aim_row+s][aim_col+t];
                    high_img_data.g_data[start_row+s][start_col+t] = ori_img_data.g_data[aim_row+s][aim_col+t];
                    high_img_data.b_data[start_row+s][start_col+t] = ori_img_data.b_data[aim_row+s][aim_col+t];
                }
            }
        }

    }
    cudaFree(d_train_r), cudaFree(d_train_g), cudaFree(d_train_b);
    cudaFree(dx);
    cudaFree(dy);
    delete []x;
    delete []y;

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
/*
void print_img_data(my_image img_data)
{
    int i = 0, j = 0, row = img_data.img_row, col = img_data.img_col;
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
            fprintf(fp, "%d ", img_data.r_data[i][j]);
        fprintf(fp, "\n");
    }
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
            fprintf(fp, "%d ", img_data.g_data[i][j]);
        fprintf(fp, "\n");
    }
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
            fprintf(fp, "%d ", img_data.b_data[i][j]);
        fprintf(fp, "\n");
    }
}
*/
