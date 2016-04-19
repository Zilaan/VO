#include "Matcher.h"
#include <time.h>
#include <opencv2/features2d/features2d.hpp>

Matcher::~Matcher()
{
}

void Matcher::computeDetectors(const Mat &image, vector<KeyPoint> &keypoints)
{
	_detector->detect(image, keypoints);
}

void Matcher::computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints,
								 Mat &descriptors)
{
	_descriptor->compute(image, keypoints, descriptors);
}

int Matcher::ratioTest(vector<vector<DMatch>> &matches)
{
	int removed = 0;

	vector<vector<DMatch>>::iterator it;

	// Check all matches
	for(it = matches.begin(); it != matches.end(); ++it)
	{
		// If a match has 2 or more NN
		if(it->size() > 1)
		{
			// Check the distance ratio
			if( (*it)[0].distance / (*it)[1].distance > _ratio )
			{
				it->clear();
				removed++;
			}
		}
		else
		{
			it->clear();
			removed++;
		}
	}
	return removed;
}

void Matcher::fastMatcher(const Mat &frame1, const Mat &frame2,
						  vector<DMatch> &good_matches, vector<KeyPoint> &keypoints)
{
	// Clear matches from previous frame
	good_matches.clear();

	// Detect features
	this->computeDetectors(frame, keypoints);

	Mat &descriptors;
	// Extract descriptors
	this->computeDescriptors(frame, keypoints, descriptors);

	vector<vector<DMatch>> matches;
	// Match 
	_matcher->knnMatch(frame1, frame2, matches, 2);
}
