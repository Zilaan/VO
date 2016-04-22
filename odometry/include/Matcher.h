#ifndef MATCHER_H_
#define MATCHER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

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
	void setDetector(const cv::Ptr<cv::FeatureDetector> &detec)
	{
		_detector = detec;
	}

	// Set feature descriptor
	void setDescriptor(const cv::Ptr<cv::DescriptorExtractor> &desc)
	{
		_descriptor = desc;
	}

	// Set feature matcher
	void setMatcher(const cv::Ptr<cv::DescriptorMatcher> &match)
	{
		_matcher = match;
	}

	// Compute the feature descriptors
	void computeDescriptors(const cv::Mat &image, cv::Mat &descriptors,
							std::vector<cv::KeyPoint> keypoints);

	// Set ratio threshold
	void setRatio(float rat)
	{
		_ratio = rat;
	}

	// Clear matches for which NN ratio is > than threshold
	int ratioTest(std::vector< std::vector<cv::DMatch> > &matches);

	// Use a fast feature matcher
	void fastMatcher(cv::Mat &prev_descriptors, cv::Mat &curr_descriptors,
					 std::vector<cv::DMatch> &good_matches);

private:
	// Pointer to feature detector object
	cv::Ptr<cv::FeatureDetector>     _detector;

	// Pointer to feature descriptor object
	cv::Ptr<cv::DescriptorExtractor> _descriptor;

	// Pointer to feature matcher
	cv::Ptr<cv::DescriptorMatcher>   _matcher;

	// Max ratio between 1st and 2nd NN
	float _ratio;
};

#endif
