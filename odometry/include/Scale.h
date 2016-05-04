#ifndef SCALE_H
#define SCALE_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

double sigma_h(std::vector<cv::Point3f> points);

double skew_gauss_kernel(std::vector<cv::Point3f> xyz);

#endif // SCALE_H
