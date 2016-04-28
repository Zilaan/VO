#include <iostream>
#include <ctime>
#include "Odometry.h"
#include "Matcher.h"
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
		fprintf(stdout, "1-2 matches: %lu\n", matches12.size());
		// 5point()
		fivePoint(f1Keypoints, f2Keypoints, matches12);
		// Triangulate()
	}
	else
	{
		// For all other frames, match features with two other frames
		mainMatcher->computeDescriptors(image, f3Descriptors, f3Keypoints);
		mainMatcher->fastMatcher(f1Descriptors, f3Descriptors, matches13);
		mainMatcher->fastMatcher(f2Descriptors, f3Descriptors, matches23);
		fprintf(stdout, "1-3 matches: %lu\n", matches13.size());
		fprintf(stdout, "2-3 matches: %lu\n\n", matches23.size());

		// Compute motion only if matches are getting to few
		if(matches13.size() < 100 || matches23.size() < 200)
		{
			sharedMatches(matches12, matches23,
						  sharedMatches12, sharedMatches23);

			sharedFeatures(f2Keypoints, f3Keypoints,
						   goodF2Key, goodF3Key, matches23);
			// Triangulate()
			triangulate(goodF2Key, goodF3Key, worldPoints);
			// P3P()
			swapAll();
			cout << "		Motion" << endl;
		}
		else
		{
			cout << "		No Motion" << endl;
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

	//cout << "size: " << x.size() << endl;

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

	// Swap projection matrices
	cM = pM.clone();
}

void Odometry::triangulate(const Mat &x,
						   const Mat &xp,
						   vector<Point3d> &X)
{
	int N = x.cols;
	Mat triang4D(1, N, CV_64F);
	Mat Rt; // Triangulated 4D points

	// Compute the current projection matrix
	hconcat(R, t, Rt);
	cM = K * Rt;

	// Triangulate points to 4D
	triangulatePoints(pM, cM, x, xp, triang4D);
	cout << "Triang comp " << endl;
	cout << triang4D.size() << endl;
	// Convert to 3D
	convertPointsHomogeneous(triang4D, X);
}

void Odometry::sharedMatches(const vector<DMatch> &m1,
							 const vector<DMatch> &m2,
							 vector<DMatch> &shared1,
							 vector<DMatch> &shared2)
{
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

void Odometry::sharedFeatures(const vector<KeyPoint> &k1,
							  const vector<KeyPoint> &k2,
							  Mat &gk1, Mat &gk2,
							  const vector<DMatch> &mask)
{
	int N = mask.size();
	int n = 0;
	gk1 = Mat(2, N, CV_64F);
	gk2 = Mat(2, N, CV_64F);
	vector<DMatch>::const_iterator it;
	for(it = mask.end(); it != mask.end(); ++it)
	{
		gk1.at<double>(1, n) = k1[it->queryIdx].pt.x;
		gk1.at<double>(2, n) = k1[it->queryIdx].pt.y;

		gk2.at<double>(1, n) = k2[it->trainIdx].pt.x;
		gk2.at<double>(2, n) = k2[it->trainIdx].pt.y;
		n++;
	}
}
