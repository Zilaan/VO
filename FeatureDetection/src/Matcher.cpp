#include "Matcher.h"
#include <time.h>
#include <opencv2/features2d/features2d.hpp>

Matcher::Matcher(parameters param) : _ratio(0.8f)
{
	// Use ORB as default detector
	_detector = ORB::create();

	// Use ORB as default descriptor
	_descriptor = ORB::create();

	// Use Brute Force with Norm Hamming as default
	_matcher = makePtr<BFMatcher>((int) NORM_HAMMING, false);
}

Matcher::~Matcher()
{
}

void Matcher::computeDescriptors(const Mat &image, Mat &descriptors)
{
	vector<KeyPoint> keypoints;
	_detector->detect(image, keypoints);
	_descriptor->compute(image, keypoints, descriptors);
}

int Matcher::ratioTest(vector< vector<DMatch> > &matches)
{
	int removed = 0;

	vector< vector<DMatch> >::iterator it;

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

void Matcher::fastMatcher(Mat &prev_descriptors, Mat &curr_descriptors,
						  vector<DMatch> &good_matches)
{
	// Clear matches from previous frame
	good_matches.clear();

	vector< vector<DMatch> > matches;
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
