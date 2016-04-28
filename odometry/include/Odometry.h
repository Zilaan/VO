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

		// Intrinsic camera paramters
		float f;
		float cu;
		float cv;

		// Ransac parameters
		int ransacIterations;
		float ransacError;
		float ransacProb;
		odometryParameters()
		{
			/*
			 * Deafault values for
			 * parameters from above
			 */

			f  = 1;
			cu = 0;
			cv = 0;

			ransacIterations = 2000;
			ransacError      = 2;
			ransacProb       = 0.99;
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
				   std::vector<cv::DMatch> &matches);

	void swapAll();

	void triangulate(const cv::Mat &x,
					 const cv::Mat &xp,
					 std::vector<cv::Point3d> &X);

	void sharedMatches(const std::vector<cv::DMatch> &m1,
					   const std::vector<cv::DMatch> &m2,
					   std::vector<cv::DMatch> &shared1,
					   std::vector<cv::DMatch> &shared2);

	void sharedFeatures(const std::vector<cv::KeyPoint> &k1,
						const std::vector<cv::KeyPoint> &k2,
						cv::Mat &gk1, cv::Mat &gk2,
						const std::vector<cv::DMatch> &mask);

	// Paramters used
	parameters param;
	cv::Mat K;
	cv::Mat E;
	cv::Mat R;
	cv::Mat t;
	cv::Mat pM;
	cv::Mat cM;
	cv::Mat goodF1Key, goodF2Key, goodF3Key;

	std::vector<cv::Point3d> worldPoints;

	Matcher *mainMatcher;

	cv::Mat f1Descriptors;
	cv::Mat f2Descriptors;
	cv::Mat f3Descriptors;

	std::vector<cv::DMatch> matches12;
	std::vector<cv::DMatch> matches13;
	std::vector<cv::DMatch> matches23;
	std::vector<cv::DMatch> sharedMatches12;
	std::vector<cv::DMatch> sharedMatches23;

	std::vector<cv::KeyPoint> f1Keypoints;
	std::vector<cv::KeyPoint> f2Keypoints;
	std::vector<cv::KeyPoint> f3Keypoints;
	//std::vector<cv::KeyPoint> goodF1;
	//std::vector<cv::KeyPoint> goodF2;
	//std::vector<cv::KeyPoint> goodF3;
	int frameNr;
};

#endif // ODOMETRY_H
