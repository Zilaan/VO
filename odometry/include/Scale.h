#ifndef SCALE_H
#define SCALE_H

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

bool sigma(const cv::Mat &points, double &sig_h);

bool gaussKernel(double &pitch, std::vector<cv::Point3d> &xyz, double &estH, double &motionTh);

#endif // SCALE_H
