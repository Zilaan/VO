#ifndef MATCHER_H_
#define MATCHER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

class Matcher
{
public:
	struct parameters
	{
		//
		int test;
		parameters()
		{
			test = 0;
		}
	};

	Matcher(parameters param);

	virtual ~Matcher();

	// Set feature detector
	void setDetector(const Ptr<FeatureDetector> &detec)
	{
		_detector = detec;
	}

	// Set feature descriptor
	void setDescriptor(const Ptr<DescriptorExtractor> &desc)
	{
		_descriptor = desc;
	}

	// Set feature matcher
	void setMatcher(const Ptr<DescriptorMatcher> &match)
	{
		_matcher = match;
	}

	// Compute the feature descriptors
	void computeDescriptors(const Mat &image, Mat &descriptors);

	// Set ratio threshold
	void setRatio(float rat)
	{
		_ratio = rat;
	}

	// Clear matches for which NN ratio is > than threshold
	int ratioTest(vector< vector<DMatch> > &matches);

	// Use a fast feature matcher
	void fastMatcher(Mat &prev_descriptors, Mat &curr_descriptors,
					 vector<DMatch> &good_matches);

private:
	// Pointer to feature detector object
	Ptr<FeatureDetector>     _detector;

	// Pointer to feature descriptor object
	Ptr<DescriptorExtractor> _descriptor;

	// Pointer to feature matcher
	Ptr<DescriptorMatcher>   _matcher;

	// Max ratio between 1st and 2nd NN
	float _ratio;
};

#endif
