#ifndef ODOMETRY_H
#define ODOMETRY_H

#include "/Users/Raman/Documents/Programmering/opencv/VO/FeatureDetection/include/Matcher.h"
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
	void estimateMotion();

	// Paramters used
	parameters param;
	Matcher *mainMatcher;

	std::vector<KeyPoint> *p_keypoints;
	std::vector<KeyPoint> *c_keypoints;
	std::vector<KeyPoint> *swap_keypoints;
	std::vector<DMatch> *good_matches;
	cv::Mat *c_descriptors;
	cv::Mat *p_descriptors;
	cv::Mat *swap_descriptors;
	bool firstRun;
};

#endif // ODOMETRY_H
