#ifndef MATCHER_H_
#define MATCHER_H_

#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

class Matcher
{
public:
	struct parameters
	{
		// Extractors Descriptors Matcher
		// 0: Fast    0: BRIEF    0: BF
		// 1: ORB     1: ORB      1: FLANN: KDTree
		// 2: SURF    2: SURF     2: FLANN: LSH
		// 3: SIFT    3: SIFT
		int extractor;
		int descriptor;
		int matcher;
		int bucketing;
		parameters()
		{
			extractor  = 1;
			descriptor = 0;
			matcher    = 2;
			bucketing  = 0;
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

	void computeFeatures(cv::Mat &image, std::vector<cv::Point2f>& points);

	// Compute the feature descriptors
	void computeDescriptors(const cv::Mat &image, cv::Mat &descriptors,
							std::vector<cv::KeyPoint> &keypoints);

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

	void featureTracking(const cv::Mat &image1, const cv::Mat &image2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2, std::vector<uchar> &status);

private:
	// Pointer to feature detector object
	cv::Ptr<cv::FeatureDetector>     _detector;

	// Pointer to feature descriptor object
	cv::Ptr<cv::DescriptorExtractor> _descriptor;

	// Pointer to feature matcher
	cv::Ptr<cv::DescriptorMatcher>   _matcher;

	// Max ratio between 1st and 2nd NN
	float _ratio;

	int bucketing;
};

#endif
