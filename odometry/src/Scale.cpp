//Functions used to calculate the relative scale
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Scale.h"
#include <stdint.h>

using namespace std;
using namespace cv;

template<class T> struct idx_cmp
{
	idx_cmp(const T arr) : arr(arr) {}
	bool operator()(const size_t a, const size_t b) const
	{
		return arr[a] < arr[b];
	}
	const T arr;
};

//Function to calculate sigma_h for the skewed Gaussian kernel
bool sigma(const Mat &points, double &sig_h)
{
	uint32_t N = (uint32_t)points.rows;
	vector<double> dist(N);
	vector<uint32_t> idx(N);
	double *d = &dist[0];
	uint32_t *i = &idx[0];
	//Point3d *p = &points[0];
	for(uint32_t n = 0; n < N; n++)
	{
		*(d++) = fabs(points.at<double>(n, 0)) + fabs(points.at<double>(n, 1)) + fabs(points.at<double>(n, 2));
		*(i++) = n;
		//p++;
	}

	sort(idx.begin(), idx.end(), idx_cmp<vector<double>&>(dist));

	// Get median
	uint32_t num_elem_half = (uint32_t)idx.size() / 2;
	sig_h = dist[idx[num_elem_half]] / 50;
	return true;
}

bool gaussKernel(double &pitch, vector<Point3d> &xyz, double &estH, double &motionTh)//function to estimate the height of the camera
{
	double sig_h;

	Mat normPoints, points; // N x 2
	Mat temp = Mat::zeros(1, 3, CV_64FC1); // 1 x 2

	// Normlize and keep points above 'ground'
	for(vector<Point3d>::iterator it = xyz.begin(); it != xyz.end(); ++it)
	{
		double x = it->x;
		double y = it->y;
		double z = it->z;
		if(z > 0) // Above ground?
		{
			temp.at<double>(0, 0) = x;
			temp.at<double>(0, 1) = y;
			temp.at<double>(0, 2) = z;
			normPoints.push_back(temp);
		}
	}

	if(normPoints.rows < 10)
		return false;

	sigma(normPoints, sig_h);
	double median = 50 * sig_h;
	if(median > motionTh)
		return false;

	double wP = 1.0 / (2.0 * sig_h * sig_h);
	sig_h = 0.01 * sig_h;
	double wM = 1.0 / (2.0 * sig_h * sig_h);
	
	temp.release();
	temp = Mat::zeros(3, 1, CV_64FC1); // 2 x 1
	temp.at<double>(0, 0) = 0;
	temp.at<double>(0, 1) = cos(-pitch);
	temp.at<double>(0, 2) = sin(-pitch);

	// Compute 'height' of all points
	points = normPoints * temp; // N x 1

	double bestSum = 0;
	uint32_t bestIdx = 0;

	uint32_t N = (uint32_t)points.rows;
	for(uint32_t i = 0; i < N; i++)
	{
		if(points.at<double>(i) > median / motionTh)
		{
			double sum = 0;
			for(uint32_t j = 0; j < N; j++)
			{
				double dist = points.at<double>(j) - points.at<double>(i);
				if (dist > 0)
					sum += exp(-dist * dist * wP);
				else
					sum += exp(-dist * dist * wM);
			}

			if(sum > bestSum)
			{
				bestSum = sum;
				bestIdx = i;
			}
		}
	}

	estH = points.at<double>(bestIdx);
	return true;
}
