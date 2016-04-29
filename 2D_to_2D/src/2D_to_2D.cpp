//2D to 2D 
#include "Matcher.h"
#include <iostream>
#include <unistd.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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
//Read image sequence
	
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
	
	Mat img; Mat prev_img; Mat match_img; Mat prev_descriptors; Mat curr_descriptors;
	vector<KeyPoint> prev_matched_keypoints; vector<KeyPoint> curr_matched_keypoints;
	vector<KeyPoint> curr_keypoints, prev_keypoints;
	vector<DMatch> good_matches;
	
	Matcher match;
//	match.setDetector("FAST");
//	match.setDescriptor("FAST");
	namedWindow("Img, q/esc to quit", CV_WINDOW_NORMAL);
	namedWindow("Prev", CV_WINDOW_NORMAL);
	namedWindow("Matches", CV_WINDOW_NORMAL);
	while( 1 )
	{	
		
		sequence >> img;
		
		if ( img.empty() || j == i ) 
		{
			cout << "Sequence finished" << endl;
			break;
		}
				
		//First image 
		if ( j == 0 )
		{
			match.fastMatcher( img, prev_keypoints, prev_descriptors );
			j++;
			prev_img = img.clone();
			imshow("Img, q/esc to quit",img);
			continue;
		}
		
		imshow("Img, q/esc to quit", img);
		imshow("Prev", prev_img);
		//waitKey(1000);
		/*char key = (char)waitKey(500);
		
		if( key == 'q' || key == 27 )
		{
			break;
		}*/
		
		
		match.fastMatcher(	img, 
					prev_keypoints, prev_descriptors, 
					curr_keypoints, curr_descriptors,
					good_matches);
		
		drawMatches( 		prev_img, prev_keypoints,
					img, curr_keypoints,
					good_matches,
					match_img);
		
		vector<Point2f> prev_matched_keypoints;
		vector<Point2f> curr_matched_keypoints;
		cout << good_matches.size() << endl;
		
		//Extract the matching points in previous and current image
		for ( vector<DMatch>::iterator it = good_matches.begin(); it != good_matches.end(); ++it )
		{
			prev_matched_keypoints.push_back(prev_keypoints[it->queryIdx].pt);
			curr_matched_keypoints.push_back(curr_keypoints[it->trainIdx].pt);
		}
		cout << prev_matched_keypoints.size() << "    " << curr_matched_keypoints.size() << endl;
		
		//Compute Essential matrix
		Mat E = findEssentialMat(	prev_matched_keypoints, curr_matched_keypoints,
						cameraMatrix, RANSAC, 0.999, 1.0, noArray() );
		Point2d pp(cam_par.cu, cam_par.cv);
		
		Mat R; Mat t;

		//Recover camera pose
		//recoverPose( 		E, prev_matched_keypoints, curr_matched_keypoints, 
					//R, t, cam_par.f, pp, mask );

		recoverPose( 		E, prev_matched_keypoints, curr_matched_keypoints,
					cameraMatrix, R, t, noArray() );		

		cout << "R: " << R << endl << " t: " << t << endl;
		//Mat R1; Mat R2; Mat t;
		//decomposeEssentialMat( 		E,R1,R2,t);
			
		//cout << "R1: " << R1 << " R2: " << R2 << " t: " << t << endl;

		imshow("Matches", match_img);
		//waitKey(3000);
		j++;
		
		//Save current info to previous
		prev_img = img.clone();
		prev_keypoints.clear();
		prev_keypoints = curr_keypoints;
		prev_descriptors = curr_descriptors.clone();
		cout << "Image: " << j << endl;
		//usleep( 100*1000 );
	}
}
