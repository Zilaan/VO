#ifndef ODOMETRY_H
#define ODOMETRY_H

#include "Matcher.h"
#include <opencv2/core/core.hpp>

class Odometry
{
public:
	struct odometryParameters
	{
		/*
		 * Some parameters used
		 * for the visual odometry
		 * i.e. focus length etc.
		 */
		int test;
		odometryParameters()
		{
			/*
			 * Deafault values for
			 * parameters from above
			 */
			test = 0;
		}
	};

	struct parameters
	{
		Odometry::odometryParameters odParam;
		Matcher::parameters maParam;
	};

	// Constructor, takes as inpute a parameter structure
	Odometry(parameters param);

	// Deconstructor
	virtual ~Odometry();

	void process(const cv::Mat &image);

private:
	// Estimate motion
	void fivePoint(const std::vector<cv::KeyPoint> &x,
				   const std::vector<cv::KeyPoint> &xp,
				   std::vector<cv::DMatch> &mask);

	void swapAll();

	// Paramters used
	parameters param;
	Matcher *mainMatcher;

	cv::Mat f1Descriptors;
	cv::Mat f2Descriptors;
	cv::Mat f3Descriptors;

	std::vector<cv::DMatch> matches12;
	std::vector<cv::DMatch> matches13;
	std::vector<cv::DMatch> matches23;

	std::vector<cv::KeyPoint> f1Keypoints;
	std::vector<cv::KeyPoint> f2Keypoints;
	std::vector<cv::KeyPoint> f3Keypoints;
	std::vector<cv::KeyPoint> goodF1;
	std::vector<cv::KeyPoint> goodF2;
	std::vector<cv::KeyPoint> goodF3;
	int frameNr;
};

#endif // ODOMETRY_H
