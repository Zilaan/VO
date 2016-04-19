#include <stdio.h>
#include <iostream>
#include <stack>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

void initModule_nonfree();
std::stack<clock_t> tictoc_stack;
double time_comb = 0;
int i;
int up_lim = 10; //upper limit in forloops
double result[10];
//functions
void readme();
void tic();
void toc();

//Main
int main( int argc, char** argv)
{
	if( argc != 2) //Check if input is correct
	{
		readme();
		return -1;
	}

	//read image and convert to grayscale	
	Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	//Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

	if( !img_1.data ) //|| !img_2.data ) //Check if image isn't empty
	{
		std::cout<< " Error reading images (!) " << std::endl;
		return -1;
	}
	
	//Reduce size of original image
	//resize(img_1, img_1, Size(), 0.1, 0.1, CV_INTER_AREA); //Resize CV_INTER_AREA when big --> small
	//CV_INTER_LINEAR(fast)/CUBIC(slow) when small --> big
	
	//Create different feature detectors
	int fastTresh = 75;	
	int orbTresh = 500;
	int surfTresh = 2395;
	int siftTresh = 500;
	int briskTresh = 30;
		
	Ptr<FastFeatureDetector> fast_det = FastFeatureDetector::create();
	Ptr<ORB> orb_det = ORB::create();
	Ptr<SURF> surf_det = SURF::create();
	Ptr<SIFT> sift_det = SIFT::create();
	Ptr<BRISK> brisk_det = BRISK::create();
	Ptr<MSER> mser_det = MSER::create(); 
	Ptr<KAZE> kaze_det = KAZE::create();
	Ptr<AKAZE> akaze_det = AKAZE::create();
	//Ptr<GoodFeaturesToTrackDetector> gftt_det = GoodFeaturesToTrackDetector::create( gfttTresh )
	//Ptr<DenseFeatureDetector> dense_det = DenseFeatureDetector::create( denseTresh );
	//Ptr<SimpleBlobDetector> sblob_det = SimpleBlobDetector::create( sblobTresh );
	//Create keypoint vectors
	std::vector<KeyPoint> kp_fast, kp_orb, kp_surf, kp_sift, kp_brisk, kp_mser, kp_kaze, kp_akaze;
	
	//Detect keypoints with different algorithms and time them over 100 iterations
	tic();
	for ( i=0;i<up_lim;i++ )
	{
		fast_det->detect( img_1, kp_fast );
	}
	toc(); result[0] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++ )
	{
		orb_det->detect( img_1, kp_orb );
	}
	toc(); result[1] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++ )
	{		
		surf_det->detect( img_1, kp_surf );
	}
	toc(); result[2] = time_comb;
	
	tic();
	for ( i=0;i<up_lim;i++ )
	{
		sift_det->detect( img_1, kp_sift );
	}
	toc(); result[3] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++)
	{
		brisk_det->detect( img_1, kp_brisk );
	}
	toc(); result[4] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++ )
	{
		mser_det->detect( img_1, kp_mser);
	}
	toc(); result[5] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++ )
	{
		kaze_det->detect( img_1, kp_kaze );
	}
	toc(); result[6] = time_comb;

	tic();
	for ( i=0;i<up_lim;i++ ) 
	{
		akaze_det->detect( img_1, kp_akaze );
	}
	toc(); result[7] = time_comb;
	
	/*//Create combined image and draw keypoints
	Mat img_fast;Mat img_orb; //Mat img_kp_2;
	
	drawKeypoints( img_1, kp_fast, img_fast, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img_1, kp_orb, img_orb, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	//drawKeypoints( img_2, kp_2, img_kp_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	*/
	//Outputfeedpack
	std::cout.precision(5);
	std::cout << "Number of FAST keypoints: " << kp_fast.size() << std::endl;
	std::cout << "Total time after " << up_lim << " iterations: " << result[0] << std::endl;
	std::cout << "Number of ORB keypoints: " << kp_orb.size() << std::endl;
	std::cout << "Total time after " << up_lim << " iterations: " << result[1] << std::endl;
	std::cout << "Number of SURF keypoints: " << kp_surf.size() << std::endl;
	std::cout << "Total time after " << up_lim << " iterations: " << result[2] << std::endl;
	std::cout << "Number of SIFT keypoints: " << kp_sift.size() << std::endl;
	std::cout << "Total time after " << up_lim << " iterations: " << result[3] << std::endl;
        std::cout << "Number of BRISK keypoints: " << kp_brisk.size() << std::endl;
        std::cout << "Total time after " << up_lim << " iterations: " << result[4] << std::endl;
        std::cout << "Number of MSER keypoints: " << kp_mser.size() << std::endl;
        std::cout << "Total time after " << up_lim << " iterations: " << result[5] << std::endl;
        std::cout << "Number of KAZE keypoints: " << kp_brisk.size() << std::endl;
        std::cout << "Total time after " << up_lim << " iterations: " << result[6] << std::endl;
        std::cout << "Number of AKAZE keypoints: " << kp_brisk.size() << std::endl;
        std::cout << "Total time after " << up_lim << " iterations: " << result[7] << std::endl;

	/*//Show detected keypoints
	imshow("FAST keypoints", img_fast);
	imshow(	"ORB keypoints", img_orb );
	waitKey(0);
	*/
	return 0;	
}

//function readme
void readme()
{
	std::cout << "Usage; ./feature_test  <img1>" << std::endl;
}

//function tic
void tic()
{
	time_comb = 0;
	tictoc_stack.push(clock());
}

//function toc
void toc()
{ 
	time_comb += time_comb + (((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC);
	tictoc_stack.pop();
}
