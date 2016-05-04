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
	K = Mat::eye(3, 3, CV_64F);
	K.at<double>(0, 0) = param.odParam.f;
	K.at<double>(0, 2) = param.odParam.cu;
	K.at<double>(1, 1) = param.odParam.f;
	K.at<double>(1, 2) = param.odParam.cv;

	// Set the initial value of R, t and projection matrix
	Mat Rt;
	R = Mat::eye(3, 3, CV_64F);
	t = Mat::zeros(3, 1, CV_64F);
	hconcat(R, t, Rt);
	pM = K * Rt;

	Tr = Mat::eye(4, 4, CV_64FC1);
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
void Odometry::process(const Mat &image)
{
	if(frameNr == 1)
	{
		// Frist frame, compute only keypoints and descriptors
		mainMatcher->computeDescriptors(image, f1Descriptors, f1Keypoints);
	}
	else if(frameNr == 2)
	{
		// Second frame available, match features with previous frame
		// and compute pose using Nister's five point
		mainMatcher->computeDescriptors(image, f2Descriptors, f2Keypoints);
		mainMatcher->fastMatcher(f1Descriptors, f2Descriptors, matches12);
		// 5point()
		fivePoint(f1Keypoints, f2Keypoints, matches12);

		// Update transformation matrix
		computeTransformation();
	}
	else
	{
		// For all other frames, match features with two other frames
		mainMatcher->computeDescriptors(image, f3Descriptors, f3Keypoints);
		mainMatcher->fastMatcher(f1Descriptors, f3Descriptors, matches13);
		mainMatcher->fastMatcher(f2Descriptors, f3Descriptors, matches23);

		// Compute motion only if matches are getting to few
		if(matches13.size() < 100 || matches23.size() < 200)
		{
			sharedMatches(matches12, matches23,
						  sharedMatches12, sharedMatches23);

			sharedFeatures(f2Keypoints, f3Keypoints,
						   goodF2Key, goodF3Key, sharedMatches23);

			triangulate(goodF2Key, goodF3Key, worldPoints);

			pnp(worldPoints, goodF3Key);

			// Update transformation matrix
			computeTransformation();

			swapAll();
		}
		else
		{
			// Throw away last frame and clear variables related to it
			f3Keypoints.clear();
			matches13.clear();
			matches23.clear();
			sharedMatches23.clear();
			goodF3Key.clear();
		}
	}
	frameNr++;
}

void Odometry::fivePoint(const vector<KeyPoint> &x,
						 const vector<KeyPoint> &xp,
						 vector<DMatch> &matches)
{
	Mat inliers;
	vector<Point2f> matchedPoints;
	vector<Point2f> matchedPointsPrime;


	// Copy only matched keypoints
	vector<DMatch>::iterator it;
	for(it = matches.begin(); it != matches.end(); ++it)
	{
		matchedPoints.push_back(x[it->queryIdx].pt);
		matchedPointsPrime.push_back(xp[it->trainIdx].pt);
	}

	E = findEssentialMat(matchedPoints, matchedPointsPrime,
						 K, RANSAC, param.odParam.ransacProb,
						 param.odParam.ransacError, inliers);

	// Recover R and t from E
	recoverPose(E, matchedPoints, matchedPointsPrime, K, R, t, inliers);

	// Update projection matrix
	computeProjection();
}

void Odometry::swapAll()
{
	// Swap descriptors
	f1Descriptors = f2Descriptors.clone();
	f2Descriptors = f3Descriptors.clone();

	// Swap keypoints
	f1Keypoints.clear();
	f1Keypoints = f2Keypoints;
	f2Keypoints.clear();
	f2Keypoints = f3Keypoints;

	// Swap matches
	matches12.clear();
	matches12 = matches23;

	sharedMatches12.clear();
	sharedMatches12 = sharedMatches23;

}

void Odometry::triangulate(const vector<Point2f> &x,
						   const vector<Point2f> &xp,
						   vector<Point3f> &X)
{
	Mat triangPt(4, x.size(), CV_64FC1);

	// Triangulate points to 4D
	triangulatePoints(pM, cM, x, xp, triangPt);

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

void Odometry::pnp(const vector<Point3f> &X,
				   const vector<Point2f> &x)
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
	computeProjection();
}

void Odometry::sharedFeatures(const vector<KeyPoint> &k1,
							  const vector<KeyPoint> &k2,
							  vector<Point2f> &gk1,
							  vector<Point2f> &gk2,
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

void Odometry::fromHomogeneous(const Mat &Pt4f, vector<Point3f> &Pt3f)
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
		Pt3f.push_back(Point3f(x, y, z));
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
