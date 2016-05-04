#ifndef SCALE_H
#define SCALE_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

double sigma(std::vector<cv::Point3f> points);

double gaussKernel(double height, std::vector<cv::Point3f> xyz);

#endif // SCALE_H
