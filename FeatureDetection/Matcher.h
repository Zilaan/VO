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
	Matcher()
	{
		// Use ORB as default detector
		_detector = ORB::create();

		// Use ORB as default descriptor
		_descriptor = ORB::create();

		// Use Brute Force with Norm Hamming as default
		_matcher = makePtr<BFMatcher>((int) NORM_HAMMING, false);
	}

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

	// Compute the feature detectors in the image
	void computeDetectors(const Mat &image, vector<KeyPoint> &keypoints);

	// Compute the feature descriptors
	void computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints,
							Mat &descriptors);

	// Set ratio threshold
	void setRatio(float rat)
	{
		_ratio = rat;
	}

	// Clear matches for which NN ratio is > than threshold
	int ratioTest(vector<vector<DMatch>> &matches);

	// Use a fast feature matcher
	void fastMatcher(const Mat &curr_frame,
					 vector<KeyPoint> &prev_keypoints, Mat &prev_descriptors,
					 vector<KeyPoint> &curr_keypoints, Mat &curr_descriptors,
					 vector<DMatch> &good_matches);

	void fastMatcher(const Mat &first_frame, vector<KeyPoint> &first_keypoints,
					 Mat &first_descriptors);

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
