
using namespace std;
using namespace cv;

Odometry::Odometry(parameters param) : param(param)
{
	mainMatcher = new Matcher(param.maParam);
}

Odometry::~Odometry()
{
}

/* Assuming that the Matcher is setup,
 * it should be implemented in the
 * constructor
 */
void Odometry::process(const Mat &image)
{
	if(firstRun)
	{
		// Only one frame available
		mainMatcher->fastMatcher(image, p_keypoints, p_descriptor);
		firstRun = false;
	}
	else
	{
		mainMatcher->fastMatcher(image, p_keypoints, p_descriptor,
								 c_keypoints, c_descriptor, good_matches);
		swap_keypoints = p_keypoints;
		p_keypoints = c_keypoints;
		c_keypoints = swap_keypoints;

		swap_descriptors = p_descriptors;
		p_descriptors = c_descriptors;
		c_descriptors = swap_descriptors;
	}
}

void Odometry::estimateMotion()
{
	// Test
	fprintf(stdout, "Feature matching is done.\n");
}
