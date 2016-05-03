//2D to 2D 
#include "../../odometry/include/Matcher.h"
#include "../../odometry/include/Odometry.h"
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

typedef struct
{
	double f;
	double cu;
	double cv;
	double height;
	double pitch;
}Camera_par;

float sigma_h(vector<Point3f> q);
float skew_gauss_kernel(float height, vector<Point3f> xyz);
void fromHomogeneous_2D(const Mat &Pt4f, vector<Point3f> &Pt3f);

int main( int argc, char** argv )
{
//Declare a few variables and construct cameraMatrix
	int i,j=0;
	Camera_par cam_par;
	cam_par.f = 645.2;
	cam_par.cu = 635.9;
	cam_par.cv = 194.1;
	cam_par.height = 1.6;
	cam_par.pitch = -0.08;
	Mat cameraMatrix = (Mat_<double>(3,3) << 	cam_par.f, 0, cam_par.cu, 
							0, cam_par.f, cam_par.cv, 
							0, 0, 1);
	Mat addMat = (Mat_<double>(1,4) << 		0,0,0,1);
	Mat Rt = Mat::zeros(3,4,CV_64F);
	Mat C = Mat::eye(4,4,CV_64F);
	Mat R = Mat::eye(3, 3, CV_64F);
	Mat t = Mat::zeros(3, 1, CV_64F);i
	Mat pos = (Mat_<double>(1,3) << 		1)
	hconcat(R, t, Rt);
	Mat pM = cameraMatrix * Rt;
	Matcher::parameters param;
	int boxWidth = 100;
	int boxHeight = 50;
	float h_est,s;
	
	//Read image sequence from input arg
	if ( argc != 2 && argc != 3 )
	{
		cout << "Call function with ./filename <path to the first image: path/img_%02d.jpg> optional: <number of images>" << endl;
		cout << argv[1] << endl;
		return -1;
	}
	
	//Init imagesequence
	string path = argv[1];
	if ( argc == 3)	{i = atoi(argv[2]);}
	else{i = -1;}
	VideoCapture sequence(path);
	if ( !sequence.isOpened() )
	{
		cerr << "Failed to open Image Sequence!" << endl;
		return -1;
	}
	//Define variables	
	Mat img, prev_img, match_img, prev_descriptors, curr_descriptors;

	vector<Point3f> XYZ;

	vector<Point2f> prev_matched_keypoints, curr_matched_keypoints;
	
	vector<KeyPoint> curr_keypoints, prev_keypoints;
	vector<KeyPoint> ground_keypoints;

	vector<DMatch> good_matches;

	//Load first image
	sequence >> img;
	
	//Construct a matcher, ORB is the default feature detector
	Matcher* match = new Matcher(param);

	//Create different windows for visual feedback
	namedWindow("Img, q/esc to quit", CV_WINDOW_NORMAL);
	namedWindow("Prev", CV_WINDOW_NORMAL);
	namedWindow("Matches", CV_WINDOW_NORMAL);
	
	//Main loop starts here
	while( 1 )
	{	

		//First image 
                if ( j == 0 )
                {
                        match->computeDescriptors ( img, prev_descriptors, prev_keypoints );
                       // match->Feature2D::detect(img, ground_keypoints, mask);
			//drawKeypoints(img, ground_keypoints,outimg,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
			j++;
                        prev_img = img.clone();
			imshow("Img, q/esc to quit",img);
			//imshow("Matches",outimg);
			//waitKey(10000);
                        continue;
                }

		//Load next image	
		sequence >> img;
	
		//Check if the sequence is finished
		if ( img.empty() || j == i ) 
		{
			cout << "Sequence finished" << endl;
			break;
		}
		
		cout << endl << "Image: " << j+1 << endl;
		//Display current image and previous image
		imshow("Img, q/esc to quit", img);
		imshow("Prev", prev_img);
		
		//Find keypoints and compute descriptors
		match->computeDescriptors( img, curr_descriptors, curr_keypoints );

		//Match current descriptors with previous descriptors
		match->fastMatcher (	prev_descriptors, curr_descriptors, good_matches );
	
		//Illustration of matches
		drawMatches( 		prev_img, prev_keypoints,
					img, curr_keypoints,
					good_matches,
					match_img);
		
		prev_matched_keypoints.clear();
		curr_matched_keypoints.clear();
		
		//Extract the matching points in previous and current image
		for ( vector<DMatch>::iterator it = good_matches.begin(); it != good_matches.end(); ++it )
		{
			prev_matched_keypoints.push_back(prev_keypoints[it->queryIdx].pt);
			curr_matched_keypoints.push_back(curr_keypoints[it->trainIdx].pt);
		}
		
		//Compute Essential matrix
		Mat E = findEssentialMat(	prev_matched_keypoints, curr_matched_keypoints,
						cameraMatrix, RANSAC, 0.999, 1.0, noArray() );
		//Calculate camera pose
		recoverPose( 		E, prev_matched_keypoints, curr_matched_keypoints,
					cameraMatrix, R, t, noArray() );		
			
		//Calculate scale
		XYZ.clear(); //empty the XYZ vector
		hconcat(R, t, Rt); //
		Mat cM = cameraMatrix * Rt;
		Mat triangPt(4,prev_matched_keypoints.size(), CV_32FC1);
		triangulatePoints(pM, cM, prev_matched_keypoints, curr_matched_keypoints, triangPt);
		fromHomogeneous_2D(triangPt, XYZ);
		
		h_est = skew_gauss_kernel(sigma_h(XYZ), XYZ);
		s = h_est/cam_par.height;
		
		//Correct the scaling error
		t = t/s;
		hconcat(R,t,Rt);
		Rt.push_back(addMat);
		//C = C*Rt.inv();
		Mat C = -R.t()*t;

		cout << "Camera position: " << endl  << C << endl;
		imshow("Matches", match_img);
		waitKey(3000);//Set delay between images here (in ms)
		j++;
		
		//Save current info to previous
		prev_img = img.clone();
		prev_keypoints.clear();
		prev_keypoints = curr_keypoints;
		prev_descriptors = curr_descriptors.clone();
	}
}

float sigma_h(vector<Point3f> q)//Function to calculate sigma_h for the skewed Gaussian kernel
{
        vector<float> res;
        if ( q.empty() ) return 0;
        else    {
                for ( vector<Point3f>::iterator it = q.begin(); it != q.end(); ++it )
                {
                        res.push_back( sqrt( pow( it->x, 2  ) + pow ( it->y, 2 ) + pow ( it->z, 2 )) );
                }

                if ( res.size() % 2 == 0 )
                        return ( 0.01 * (res[res.size()/2 - 1] + res[res.size()/2]) );
                else
                        return ( 0.02 * res[res.size()/2] );
        }
}


float skew_gauss_kernel(float height, vector<Point3f> xyz)//function to estimate the height of the camera
{
        float sig_h = sigma_h( xyz );
        vector<float> val;
        vector<float>::iterator pos;

        for( vector<Point3f>::iterator it = xyz.begin(); it != xyz.end(); ++it )
        {
                if ( height - it->y > 0)
                        val.push_back( exp( (-0.5*pow( it->y, 2 ) )/pow( sig_h, 2 ) ) );
                else
                        val.push_back( exp( (-0.5*pow( it->y,2 ) )/pow( sig_h*0.01, 2 ) ) );
        }
        pos = max_element ( val.begin(), val.end() );

        return *pos;
}

void fromHomogeneous_2D(const Mat &Pt4f, vector<Point3f> &Pt3f)
{
        Pt3f.clear();
        int N = Pt4f.cols; // Number of 4-channel elements
        float x, y, z, w;
        for(int i = 0; i < N; i++)
        {
                // Convert the points to Euclidean space
                w = Pt4f.at<float>(3, i);
                z = Pt4f.at<float>(2, i) / w;
                y = Pt4f.at<float>(1, i) / w;
                x = Pt4f.at<float>(0, i) / w;
                Pt3f.push_back(Point3f(x, y, z));
        }
}
