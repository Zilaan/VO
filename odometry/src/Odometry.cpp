#include <iostream>
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
		E = fivePoint(f1Keypoints, f2Keypoints, matches12);
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
			// Triangulate()
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

Mat Odometry::fivePoint(const vector<KeyPoint> &x,
						const vector<KeyPoint> &xp,
						vector<DMatch> &matches)
{
	Mat essential, inliers, R, t;
	vector<Point2f> matchedPoints;
	vector<Point2f> matchedPointsPrime;

	cout << "size: " << x.size() << endl;

	// Copy only matched keypoints
	vector<DMatch>::iterator it;
	for(it = matches.begin(); it != matches.end(); ++it)
	{
		matchedPoints.push_back(x[it->queryIdx].pt);
		matchedPointsPrime.push_back(xp[it->trainIdx].pt);
	}

	essential = findEssentialMat(matchedPoints, matchedPointsPrime,
								 K, RANSAC, param.odParam.ransacProb,
								 param.odParam.ransacError, inliers);

	// Recover R and t from E
	recoverPose(essential, matchedPoints, matchedPointsPrime, K, R, t, inliers);
	cout << "R: " << R << endl;
	cout << "t: " << t << endl;
	return essential;
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
}

void Odometry::triangulate(const Mat &M1, const Mat &M2,
			const vector<KeyPoint> &x,
			const vector<KeyPoint> &xp,
			vector<Point3d> &X)
{
	Mat triang4D; // Triangulated 4D points

	// Triangulate points to 4D
	triangulatePoints(M1, M2, x, xp, triang4D);

	// Convert to 3D
	convertPointsHomogeneous(triang4D, X);
}
