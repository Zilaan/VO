#include <iostream>
#include <ctime>
#include "Odometry.h"
#include "Matcher.h"
#include "Scale.h"
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

Odometry::Odometry(parameters param) : param(param), frameNr(1)
{
	mainMatcher = new Matcher(param.maParam);

	// Set the intrinsic camera paramters
	K = Mat::eye(3, 3, CV_64FC1);
	K.at<double>(0, 0) = param.odParam.f;
	K.at<double>(0, 2) = param.odParam.cu;
	K.at<double>(1, 1) = param.odParam.f;
	K.at<double>(1, 2) = param.odParam.cv;

	// Set the initial value of R, t and projection matrix
	Mat Rt;
	R = Mat::eye(3, 3, CV_64FC1);
	t = Mat::zeros(3, 1, CV_64FC1);
	RFinal = R.clone();
	tFinal = t.clone();
	hconcat(R, t, Rt);
	pM = K * Rt;

	//Tr = Mat::eye(4, 4, CV_64FC1);
}

Odometry::~Odometry()
{
	// Free memory allocated by mainMatcher
	delete mainMatcher;
}

/* Assuming that the Matcher is setup,
 * it should be implemented in the
 * constructor
 */
bool Odometry::process(const Mat &image)
{
	if(frameNr == 1)
	{
		// Frist frame, compute only keypoints and descriptors
		mainMatcher->computeDescriptors(image, f1Descriptors, f1Keypoints);
	}
	else
	{
		// Second frame available, match features with previous frame
		// and compute pose using Nister's five point
		mainMatcher->computeDescriptors(image, f2Descriptors, f2Keypoints);
		mainMatcher->fastMatcher(f1Descriptors, f2Descriptors, matches12);

		// Compute R and t
		fivePoint(f1Keypoints, f2Keypoints, matches12);

		// Update projection matrix i.e. cM
		computeProjection();

		// Compute 3D points
		triangulate(pMatchedPoints, cMatchedPoints, worldPoints);

		// Compute scale
		correctScale(worldPoints);

		pM.release();
		pM = cM.clone();

		tFinal = tFinal + RFinal * t;
		RFinal = R * RFinal;

		//fprintf(stdout, "X:%4.2f, Y:%4.2f, Z:%4.2f\n", tFinal.at<double>(0, 0), tFinal.at<double>(0, 1), tFinal.at<double>(0, 2));

		swapAll();


	}
	frameNr++;
	return true;
}

void Odometry::fivePoint(const vector<KeyPoint> &xp,
						 const vector<KeyPoint> &x,
						 vector<DMatch> &matches)
{
	Mat inliers;

	pMatchedPoints.clear();
	cMatchedPoints.clear();

	// Copy only matched keypoints
	vector<DMatch>::iterator it;
	for(it = matches.begin(); it != matches.end(); ++it)
	{
		pMatchedPoints.push_back(xp[it->queryIdx].pt);
		cMatchedPoints.push_back(x[it->trainIdx].pt);
	}

	E = findEssentialMat(pMatchedPoints, cMatchedPoints,
						 K, RANSAC, param.odParam.ransacProb,
						 param.odParam.ransacError, inliers);

	// Recover R and t from E
	recoverPose(E, pMatchedPoints, cMatchedPoints, K, R, t, inliers);
}

void Odometry::swapAll()
{
	// Swap descriptors
	f1Descriptors = f2Descriptors.clone();
	//f2Descriptors = f3Descriptors.clone();

	// Swap keypoints
	f1Keypoints.clear();
	f1Keypoints = f2Keypoints;
	f2Keypoints.clear();
	//f2Keypoints = f3Keypoints;

	// Swap matches
	matches12.clear();
	//matches12 = matches23;

	//sharedMatches12.clear();
	//sharedMatches12 = sharedMatches23;

}

void Odometry::triangulate(const vector<Point2d> &xp,
						   const vector<Point2d> &x,
						   vector<Point3d> &X)
{
	Mat triangPt(4, x.size(), CV_64FC1);

	// Triangulate points to 4D
	triangulatePoints(pM, cM, xp, x, triangPt);

	fromHomogeneous(triangPt, X);
}

void Odometry::sharedMatches(const vector<DMatch> &m1,
							 const vector<DMatch> &m2,
							 vector<DMatch> &shared1,
							 vector<DMatch> &shared2)
{
	shared1.clear();
	shared2.clear();
	vector<DMatch>::const_iterator it2, it1;
	for(it1 = m1.begin(); it1 != m1.end(); ++it1)
	{
		for(it2 = m2.begin(); it2 != m2.end(); ++it2)
		{
			if(it2->queryIdx == it1->trainIdx)
			{
				shared1.push_back(*it1);
				shared2.push_back(*it2);
				break;
			}
		}
	}
}

void Odometry::pnp(const vector<Point3d> &X,
				   const vector<Point2d> &x)
{
	Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
	Mat rvec = Mat::zeros(3, 1, CV_64FC1); // output rotation vector
	Mat tvec = Mat::zeros(3, 1, CV_64FC1); // output translation vector
	bool useExtrinsicGuess = false;

	solvePnPRansac(X, x, K, distCoeffs, rvec, tvec, useExtrinsicGuess,
				   param.odParam.ransacIterations,
				   param.odParam.ransacError,
				   param.odParam.ransacProb,
				   noArray(),
				   param.odParam.pnpFlags);

	Rodrigues(rvec, R); // Convert rotation vector to matrix
	t = tvec;

	// Update projection matrix
	pM.release();
	pM = cM.clone();
}

void Odometry::sharedFeatures(const vector<KeyPoint> &k1,
							  const vector<KeyPoint> &k2,
							  vector<Point2d> &gk1,
							  vector<Point2d> &gk2,
							  const vector<DMatch> &mask)
{
	gk1.clear();
	gk2.clear();
	vector<DMatch>::const_iterator it;
	for(it = mask.begin(); it != mask.end(); ++it)
	{
		gk1.push_back(k1[it->queryIdx].pt);
		gk2.push_back(k2[it->trainIdx].pt);
	}
}

void Odometry::fromHomogeneous(const Mat &Pt4f, vector<Point3d> &Pt3f)
{
	Pt3f.clear();
	int N = Pt4f.cols; // Number of 4-channel elements
	double x, y, z, w;
	for(int i = 0; i < N; i++)
	{
		// Convert the points to Euclidean space
		w = Pt4f.at<double>(3, i);
		z = Pt4f.at<double>(2, i) / w;
		y = Pt4f.at<double>(1, i) / w;
		x = Pt4f.at<double>(0, i) / w;
		Pt3f.push_back(Point3d(x, y, z));
	}
}

void Odometry::computeTransformation()
{
	for(int r = 0; r < cM.rows; r++)
	{
		for(int c = 0; c < cM.cols; c++)
		{
			Tr.at<double>(r, c) = cM.at<double>(r, c);
		}
	}
}

void Odometry::computeProjection()
{
	Mat Rt;

	// Compute the current projection matrix
	hconcat(R, t, Rt);
	cM = K * Rt;
}

void Odometry::correctScale(vector<Point3d> &points)
{
	double trueHeight = param.odParam.cameraHeight;

	// Compute estimated height
	double estHeight = gaussKernel(trueHeight, points);

	// Compute the scaling factor
	double rho = trueHeight / estHeight;

	fprintf(stdout, "estHeight:%4.2f\n", estHeight);
	//t = t * rho;
}
