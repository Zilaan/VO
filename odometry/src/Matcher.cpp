#include "Matcher.h"
#include <time.h>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

Matcher::Matcher(parameters param) : _ratio(0.8f)
{
	Ptr<FeatureDetector> orb = ORB::create();
//	Ptr<FeatureDetector> orb = ORB::create(
//			 40,				// nFeatures
//			 1.7f,				// scaleFactor
//			 8,					// nlevels
//			 7,						// edgeThreshold
//			 0,					// firstLevel
//			 2,					// WTA_K
//			 ORB::HARRIS_SCORE, // scoreType
//			 7,						// patchSize
//			 20					// fastThreshold
//);
	// Use ORB as default detector
	_detector = orb;
	// Use ORB as default descriptor
	_descriptor = orb;

	Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
	Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

	// instantiate FlannBased matcher
	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
	_matcher = matcher;

	//// Use ORB as default detector
	//_detector = ORB::create();

	//// Use ORB as default descriptor
	//_descriptor = ORB::create();

	//// Use Brute Force with Norm Hamming as default
	//_matcher = makePtr<BFMatcher>((int) NORM_HAMMING, false);
}

Matcher::~Matcher()
{
}

void Matcher::computeDescriptors(const Mat &image, Mat &descriptors,
								 vector<KeyPoint> &keypoints)
{
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
