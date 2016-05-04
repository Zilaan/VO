#ifndef ODOMETRY_H
#define ODOMETRY_H

#include "Matcher.h"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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
		double f;
		double cu;
		double cv;
		double cameraHeight;

		// Ransac parameters
		int pnpFlags;
		int ransacIterations;
		double ransacError;
		double ransacProb;
		odometryParameters()
		{
			/*
			 * Deafault values for
			 * parameters from above
			 */

			f  = 645.2;
			cu = 635.9;
			cv = 194.1;
			cameraHeight = 1.6;

			pnpFlags         = cv::SOLVEPNP_P3P;
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

	cv::Mat getMotion()
	{
		return tFinal;
	}

private:
	// Estimate motion
	void fivePoint(const std::vector<cv::KeyPoint> &x,
				   const std::vector<cv::KeyPoint> &xp,
				   std::vector<cv::DMatch> &matches);

	void swapAll();

	void triangulate(const std::vector<cv::Point2d> &xp,
					 const std::vector<cv::Point2d> &x,
					 std::vector<cv::Point3d> &X);

	void sharedMatches(const std::vector<cv::DMatch> &m1,
					   const std::vector<cv::DMatch> &m2,
					   std::vector<cv::DMatch> &shared1,
					   std::vector<cv::DMatch> &shared2);

	void pnp(const std::vector<cv::Point3d> &X,
			 const std::vector<cv::Point2d> &x);

	void sharedFeatures(const std::vector<cv::KeyPoint> &k1,
						const std::vector<cv::KeyPoint> &k2,
						std::vector<cv::Point2d> &gk1,
						std::vector<cv::Point2d> &gk2,
						const std::vector<cv::DMatch> &mask);

	void fromHomogeneous(const cv::Mat &Pt4f, std::vector<cv::Point3d> &Pt3f);

	void computeTransformation();

	void computeProjection();

	void correctScale(std::vector<cv::Point3d> &points);

	// Matcher object
	Matcher *mainMatcher;

	// Paramters used
	parameters param;

	cv::Mat K;  // Intrisic parameters for camera
	cv::Mat E;  // Essential matrix
	cv::Mat RFinal;  // Rotation matrix
	cv::Mat tFinal;  // Translation vector
	cv::Mat R;  // Rotation matrix
	cv::Mat t;  // Translation vector
	cv::Mat pM; // Previous projection matrix
	cv::Mat cM; // Current projection matrix
	cv::Mat Tr; // Previous and current trans matrix

	// Keypoints filtered with shared matches
	std::vector<cv::Point2d> goodF1Key, goodF2Key, goodF3Key;

	// Triangulated Euclidean points
	std::vector<cv::Point3d> worldPoints;

	// Descriptors from the three frames
	cv::Mat f1Descriptors;
	cv::Mat f2Descriptors;
	cv::Mat f3Descriptors;

	// Matches from three frames
	std::vector<cv::DMatch> matches12;
	std::vector<cv::DMatch> matches13;
	std::vector<cv::DMatch> matches23;

	// Shared matches
	std::vector<cv::DMatch> sharedMatches12;
	std::vector<cv::DMatch> sharedMatches23;

	// Keypoints from the three frames
	std::vector<cv::KeyPoint> f1Keypoints;
	std::vector<cv::KeyPoint> f2Keypoints;
	std::vector<cv::KeyPoint> f3Keypoints;

	std::vector<cv::Point3d> X12;
	std::vector<cv::Point3d> X13;
	std::vector<cv::Point3d> X23;

	std::vector<cv::Point2d> pMatchedPoints;
	std::vector<cv::Point2d> cMatchedPoints;
	int frameNr;
};

#endif // ODOMETRY_H
