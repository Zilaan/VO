#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


int main(int argc, char** argv)
{
	if ( argc != 3 ) //See if there are 3 arguments
	{
		printf("Error, number of arguments should be 3\n");
		return -1;
	}
	
	cv::Mat image1, image2; //Read the two images
	image1 = cv::imread( argv[1], 0 );
	image2 = cv::imread( argv[2], 0 );

	if ( !image1.data || !image2.data ) //If inputs are not images, quit
	{
		printf("No image data \n");
		return -1;
	}

	cv::resize(image1, image1, cv::Size(), 0.1, 0.1, CV_INTER_AREA); // Resize CV_INTER_AREA when big --> small
	cv::resize(image2, image2, cv::Size(), 0.1, 0.1, CV_INTER_AREA); //CV_INTER_LINEAR(fast)/CUBIC(slow) 
	
	int nr = 100;

	cv::Ptr<cv::FAST> detector = cv::FAST::create( nr );
	
	std::vector<cv::KeyPoint> kp1, kp2;
	
	detector->detect( image1, kp1 );
	detector->detect( image2, kp2 );

	cv::Mat img_kp_1; cv::Mat img_kp_2;

	cv::drawKeypoints( image1, kp1, img_kp_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	cv::drawKeypoints( image2, kp2, img_kp_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	
	cv::imshow("kp1", img_kp_1);
	cv::imshow("kp2", img_kp_2);

	cv::waitKey(0);	
	/*cv::namedWindow("Image one resized", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image one resized", image1);

	cv::namedWindow("Image two resized", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image two resized", image2); 
	
	cv::waitKey(0);
	*/
	return 0;
}
