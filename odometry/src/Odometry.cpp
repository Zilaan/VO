#include <iostream>
#include "Odometry.h"
#include "Matcher.h"
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

Odometry::Odometry(parameters param) : param(param), frameNr(1)
{
	mainMatcher = new Matcher(param.maParam);
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
		fprintf(stdout, "Matches: %lu\n", matches12.size());
		// 5point()
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

void Odometry::fivePoint(const vector<KeyPoint> &x,
						const vector<KeyPoint> &xp,
						vector<DMatch> &mask)
{
	vector<Point2f> matchedPoints;
	vector<Point2f> matchedPointsPrime;

	vector<DMatch>::iterator it;
	for(it = mask.begin(); it != mask.end(); ++it)
	{
		;
	}
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
