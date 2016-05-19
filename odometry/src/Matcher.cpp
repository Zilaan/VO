#include "Matcher.h"
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Matcher::Matcher(parameters param) : _ratio(0.8f)
{
// Extractors Descriptors Matcher
// 0: Fast    0: BRIEF    0: BF
// 1: ORB     1: ORB      1: FLANN: KDTree
// 2: SURF    2: SURF     2: FLANN: LSH
// 3: SIFT    3: SIFT

//	Ptr<FeatureDetector> orb = ORB::create(
//			 75,				// nFeatures
//			 1.7f,				// scaleFactor
//			 8,					// nlevels
//			 7,						// edgeThreshold
//			 0,					// firstLevel
//			 2,					// WTA_K
//			 ORB::HARRIS_SCORE, // scoreType
//			 7,						// patchSize
//			 20					// fastThreshold
//);
	Ptr<FeatureDetector> ext;
	switch (param.extractor)
	{
		case 0:
			ext =  FastFeatureDetector::create(20, true, FastFeatureDetector::TYPE_9_16);
			break;

		case 1:
			ext = ORB::create();
			break;

		case 2:
			ext = SURF::create();
			break;

		case 3:
			ext = SIFT::create();
			break;
	}

	Ptr<FeatureDetector> des;
	switch (param.descriptor)
	{
		case 0:
			des = BriefDescriptorExtractor::create();
			break;

		case 1:
			des = ORB::create();
			break;

		case 2:
			des = SURF::create();
			break;

		case 3:
			des = SIFT::create();
			break;
	}

	Ptr<flann::IndexParams> indexParams;
	Ptr<flann::SearchParams> searchParams;
	Ptr<DescriptorMatcher> match;
	switch (param.matcher)
	{
		case 0:
			match = makePtr<BFMatcher>((int) NORM_HAMMING, false);
			break;

		case 1:
			indexParams  = makePtr<flann::KDTreeIndexParams>();
			searchParams = makePtr<flann::SearchParams>(50);
			match        = makePtr<FlannBasedMatcher>(indexParams, searchParams);
			break;

		case 2:
			indexParams  = makePtr<flann::LshIndexParams>(6, 12, 1);
			searchParams = makePtr<flann::SearchParams>(50);
			match        = makePtr<FlannBasedMatcher>(indexParams, searchParams);
			break;
	}

	_detector   = ext;     // Set the extractor
	_descriptor = des;     // Set the descriptor
	_matcher    = match;   // Set the matcher

	bucketing = param.bucketing;
}

Matcher::~Matcher()
{
}

bool Matcher::computeFeatures(Mat &image, vector<Point2f>& points)
{
	points.clear();
	vector<KeyPoint> keypoints;
	_detector->detect(image, keypoints);
	if(keypoints.size() < 10)
		return false;
	KeyPoint::convert(keypoints, points, vector<int>());

	return true;
}

bool Matcher::computeDescriptors(const Mat &image, Mat &descriptors,
								 vector<KeyPoint> &keypoints)
{
	int y[] = {0, 188};
	int yd = 188;
	int x[] = {0, 310, 620, 930};
	int xd = 310;
	vector<KeyPoint> tempKey;

	if(bucketing == 1)
	{
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				tempKey.clear();
				Mat mask = Mat::zeros(image.size(), CV_8UC1);
				Mat roi(mask, Rect(x[j], y[i], xd, yd));
				roi = Scalar(255);
				_detector->detect(image, tempKey, mask);
				//_detector->detect(image, keypoints);
				//cout << endl << "tempKey" << endl;;
				//for(vector<KeyPoint>::iterator it = tempKey.begin(); it != tempKey.end(); ++it)
				//{
				//	//it->pt.x = it->pt.x + x[j];
				//	//it->pt.y = it->pt.y + y[i];
				//	//it->pt.x = it->pt.x + 10;
				//	//it->pt.y = it->pt.y + 10;
				//	cout << it->pt << endl;
				//}
				keypoints.insert(keypoints.end(), tempKey.begin(), tempKey.end());
				if(keypoints.size() < 10)
					return false;
			}
		}
	}
	else
	{
		_detector->detect(image, keypoints);
		if(keypoints.size() < 10)
			return false;
	}
	_descriptor->compute(image, keypoints, descriptors);

	return true;
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

bool Matcher::fastMatcher(Mat &prev_descriptors, Mat &curr_descriptors,
						  vector<DMatch> &good_matches)
{
	// Clear matches from previous frame
	good_matches.clear();

	vector< vector<DMatch> > matches;
	_matcher->knnMatch(prev_descriptors, curr_descriptors, matches, 2);
	if(matches.size() < 10)
		return false;

	// Performe ratio test on the matches
	ratioTest(matches);

	// Fill the good matches into correct container
	vector<vector<DMatch> >::iterator it;
	for (it = matches.begin(); it != matches.end(); ++it)
	{
		if (!it->empty())
			good_matches.push_back((*it)[0]);
	}

	if(good_matches.size() < 10)
		return false;

	return true;
}

bool Matcher::featureTracking(const Mat &image1, const Mat &image2, vector<Point2f> &points1, vector<Point2f> &points2, vector<uchar> &status)
{
	vector<float> err;
	Size winSize = Size(21, 21);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

	calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

	//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
	int indexCorrection = 0;
	for( int i = 0; i < status.size(); i++)
	{
		Point2f pt = points2.at(i - indexCorrection);
		if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
		{
			if((pt.x < 0) || (pt.y < 0))
			{
				status.at(i) = 0;
			}
			points1.erase (points1.begin() + (i - indexCorrection));
			points2.erase (points2.begin() + (i - indexCorrection));
			indexCorrection++;
		}
	}

	if(points1.size() < 10)
		return false;

	return true;
}
