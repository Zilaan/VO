#include "Matcher.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	// Instantiate Matcher
	Matcher fastMatcher;

	Ptr<FeatureDetector> orb = ORB::create();

	// Set feature detector
	fastMatcher.setDetector(orb);

	// Set feature descriptor
	fastMatcher.setDescriptor(orb);

	// Instantiate LSH index and FLANN search
	Ptr<flann::IndexParams> indexParams = makePtr<flann::LshIndexParams>(6, 12, 1);
	Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(50);

	// Instantiate FlannBased matcher
	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

	// Set feature matcher
	fastMatcher.setDescriptor(matcher);

	return 0;
}
