#ifndef ODOMETRY_H
#define ODOMETRY_H

#include "Matcher.h"
#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <sstream>
#include <fstream>

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
		double pitch;

		// Ransac parameters
		int pnpFlags;
		int ransacIterations;
		double ransacError;
		double ransacProb;
		int scaling;
		double motionThreshold;
		int method;
		int doBundle;
		int bundleParam;
		int imageSequence;
		odometryParameters()
		{
			/*
			 * Deafault values for
			 * parameters from above
			 */

			f  = 718.856;
			cu = 607.1928;
			cv = 185.2157;
			cameraHeight = 1.6;
			pitch = -0.08;

			pnpFlags         = cv::SOLVEPNP_P3P;
			ransacIterations = 2000;
			ransacError      = 1;
			ransacProb       = 0.999;
			scaling          = 2;
			motionThreshold  = 100;
			method           = 1;
			doBundle         = 1;
			bundleParam      = 500;
			imageSequence    = 0;
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

	bool process(const cv::Mat &image);

	cv::Mat getMotion()
	{
		return Tr_delta;
	}

	std::vector<cv::Point2d> getKeypoints(int a)
	{
		if(a == 0)
			return pMatchedPoints;
		else
			return cMatchedPoints;
	}

	double getHeight()
	{
		return estHeight;
	}

	int32_t getNumKeypoints()
	{
		if(param.odParam.method == 0)
			return (int32_t) f1Keypoints.size();
		else
			return (int32_t) status.size();
	}

	int32_t getNumInliers()
	{
		return (int32_t) sum(inliers)[0];
	}

	int32_t getNumMatches()
	{
		if(param.odParam.method == 0)
			return (int32_t) matches12.size();
		else
		{
			if(frameNr <= 2)
				return (int32_t) f1Points.size();
			else
				return (int32_t) f2Points.size();
		}
	}

private:
	// Estimate motion
	void fivePoint(const std::vector<cv::KeyPoint> &xp,
				   const std::vector<cv::KeyPoint> &x,
				   std::vector<cv::DMatch> &matches);

	void fivePoint(const std::vector<cv::Point2f> &xp,
				   const std::vector<cv::Point2f> &x);

	void swapAll();

	bool triangulate(const std::vector<cv::Point2d> &xp,
					 const std::vector<cv::Point2d> &x,
					 std::vector<cv::Point3d> &X);

	void pnp(const std::vector<cv::Point3d> &X,
			 const std::vector<cv::Point2d> &x);

	void fromHomogeneous(const cv::Mat &Pt4f, std::vector<cv::Point3d> &Pt3f);

	std::vector<double> transformationVec(const cv::Mat &RMat, const cv::Mat &tvec);

	cv::Mat transformationMat(const std::vector<double> &tr);

	void correctScale(std::vector<cv::Point3d> &points);

	bool getTrueScale(int frame_id);

	void sharedPoints(const cv::Mat &inl, const cv::Mat &inl2, const std::vector<uchar> &s23);

	std::vector<double> bundle();

	void sharedPoints(const std::vector<cv::DMatch> &m1,
					  const std::vector<cv::DMatch> &m2);

	// Matcher object
	Matcher *mainMatcher;

	// Paramters used
	parameters param;

	cv::Mat K;  // Intrisic parameters for camera
	cv::Mat E;  // Essential matrix
	cv::Mat R;  // Rotation matrix
	cv::Mat prevR;  // Rotation matrix
	cv::Mat t;  // Translation vector
	cv::Mat prevT;  // Translation vector
	cv::Mat Tr_delta; // Previous and current trans matrix
	double rho; // Scale factor
	double estHeight; // Estimated height by kernel

	// Triangulated Euclidean points
	std::vector<cv::Point3d> worldPoints;

	// Descriptors from the three frames
	cv::Mat f1Descriptors;
	cv::Mat f2Descriptors;
	cv::Mat f3Descriptors;
	cv::Mat inliers;
	cv::Mat inliers1;
	cv::Mat inliers2;
	cv::Mat prevImage;

	// Matches from three frames
	std::vector<cv::DMatch> matches12;
	std::vector<cv::DMatch> matches13;
	std::vector<cv::DMatch> matches23;

	// Keypoints from the three frames
	std::vector<cv::KeyPoint> f1Keypoints;
	std::vector<cv::Point2f> f1Points;
	std::vector<cv::KeyPoint> f2Keypoints;
	std::vector<cv::Point2f> f2Points;
	std::vector<cv::KeyPoint> f3Keypoints;
	std::vector<cv::Point2f> f3Points;
	std::vector<uchar> status;

	std::vector<cv::Point2d> f1Double;
	std::vector<cv::Point2d> f2Double;
	std::vector<cv::Point2d> f3Double;
	std::vector<cv::Point3d> TriangPoints;

	std::vector<cv::Point2d> pMatchedPoints;
	std::vector<cv::Point2d> cMatchedPoints;
	int frameNr;
	
	bool retrack;
	int bundleAdj;
	int optimParam;
};

#endif // ODOMETRY_H
