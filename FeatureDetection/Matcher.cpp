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

void fastMatcher(const Mat &curr_frame,
				 vector<KeyPoint> &prev_keypoints, Mat &prev_descriptors,
				 vector<KeyPoint> &curr_keypoints, Mat &curr_descriptors,
				 vector<DMatch> &good_matches)
{
	// Clear matches from previous frame
	good_matches.clear();

	// Detect features
	this->computeDetectors(curr_frame, curr_keypoints);

	// Extract descriptors
	this->computeDescriptors(curr_frame, curr_keypoints, curr_descriptors);

	vector<vector<DMatch>> matches;
	// Match 
	_matcher->knnMatch(prev_descriptors, curr_descriptors, matches, 2);

	// Performe ratio test on the matches
	ratioTest(matches);

	// Fill the good matches into correct container
	vector<vector<DMatch> >::iterator it;
	for (it = matches.begin(); it != matches.end(); ++it)
	{
		if (!it->empty())
			good_matches.push_back((*it)[0]);
	}
}

void fastMatcher(const Mat &first_frame, vector<KeyPoint> &first_keypoints,
				 Mat &first_descriptors)
{
	// Detect features
	this->computeDetectors(first_frame, first_keypoints);

	// Extract descriptors
	this->computeDescriptors(first_frame, first_keypoints, first_descriptors);
}
