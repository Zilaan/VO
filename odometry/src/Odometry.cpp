#include <iostream>
#include <ctime>
#include "Odometry.h"
#include "Matcher.h"
#include "Scale.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

Odometry::Odometry(parameters param) : param(param), frameNr(1)
{
	mainMatcher = new Matcher(param.maParam);

	// Set the intrinsic camera paramters
	K = Mat::eye(3, 3, CV_64FC1);
	K.at<double>(0, 0) = param.odParam.f;
	K.at<double>(0, 2) = param.odParam.cu;
	K.at<double>(1, 1) = param.odParam.f;
	K.at<double>(1, 2) = param.odParam.cv;

	rho = 1.0;
}

Odometry::~Odometry()
{
	// Free memory allocated by mainMatcher
	delete mainMatcher;
}

bool Odometry::process(const Mat &image)
{
	if(param.odParam.method == 0) // Use matcher
	{
		if(frameNr == 1)
		{
			// Frist frame, compute only keypoints and descriptors
			if(!mainMatcher->computeDescriptors(image, f1Descriptors, f1Keypoints))
				return false;
		}
		else
		{
			// Second frame available, match features with previous frame
			// and compute pose using Nister's five point
			if(!mainMatcher->computeDescriptors(image, f2Descriptors, f2Keypoints))
				return false;
			if(!mainMatcher->fastMatcher(f1Descriptors, f2Descriptors, matches12))
				return false;

			// Compute R and t
			fivePoint(f1Keypoints, f2Keypoints, matches12);

			// Compute 3D points
			triangulate(pMatchedPoints, cMatchedPoints, worldPoints);

			// Compute scale
			if(param.odParam.scaling == 1)
				correctScale(worldPoints);
			else if (param.odParam.scaling == 2)
				getTrueScale(frameNr);
			else
				;

			vector<double> tr_delta = transformationVec(R, t);

			Tr_delta = transformationMat(tr_delta);

			swapAll();
		}
	}
	else // Use tracker
	{
		if(frameNr == 1)
		{
			// Frist frame, compute only keypoints and descriptors
			if(!mainMatcher->computeFeatures(image, f1Points))
				return false;
			prevImage = image.clone();
		}
		else if(frameNr == 2)
		{
			// Second frame available, match features with previous frame
			// and compute pose using Nister's five point
			if(!mainMatcher->featureTracking(prevImage, image, f1Points, f2Points, status))
				return false;

			// Compute R and t
			fivePoint(f1Points, f2Points);
			if(pMatchedPoints.size() < 100)
				return false;

			// Compute 3D points
			triangulate(pMatchedPoints, cMatchedPoints, worldPoints);

			// Compute scale
			if(param.odParam.scaling == 1)
				correctScale(worldPoints);
			else if (param.odParam.scaling == 2)
				getTrueScale(frameNr);
			else
				;

			vector<double> tr_delta = transformationVec(R, t);

			Tr_delta = transformationMat(tr_delta);

			if(f1Points.size() < 1000)
			{
				if(!mainMatcher->computeFeatures(prevImage, f1Points))
					return false;
				if(!mainMatcher->featureTracking(prevImage, image, f1Points, f2Points, status))
					return false;
			}
			prevImage = image.clone();
		}
		else
		{
			if(!mainMatcher->featureTracking(prevImage, image, f2Points, f3Points, status))
				return false;

			// Get shared features
			if(frameNr == 3)
			{
				for(int i = 0; i < status.size(); i++)
				{
					if(status.at(i) == 1 && inliers.at<uint8_t>(i) == 1)
					{
						f1Double.push_back(pMatchedPoints[i]);
						f2Double.push_back(cMatchedPoints[i]);
						f3Double.push_back(Point2d(f3Points[i].x, f3Points[i].y));
						TriangPoints.push_back(worldPoints[i]);
					}
				}
			}
			else
			{
				f1Double.clear();
				f2Double.clear();
				f3Double.clear();
				TriangPoints.clear();
				cout << "Rows: " << inliers.rows << endl;
				uint32_t idx;
				for(int i = 0; i < inliers.rows; i++)

					idx = inliers.at<uint8_t>(i);
					if(status.at(idx) == 1)
					{
						f1Double.push_back(Point2d(f1Points[idx].x, f1Points[idx].y));
						f2Double.push_back(Point2d(f2Points[idx].x, f2Points[idx].y));
						f3Double.push_back(Point2d(f3Points[i].x, f3Points[i].y));
						TriangPoints.push_back(worldPoints[i]);
					}
				}
			}

			pnp(TriangPoints, f3Double);
			// Bunlde adjustment now or later?
			triangulate(f2Double, f3Double, worldPoints);

			vector<double> tr_delta = transformationVec(R, t);

			Tr_delta = transformationMat(tr_delta);

			if(f1Points.size() < 1000)
			{
				if(!mainMatcher->computeFeatures(prevImage, f1Points))
					return false;
				if(!mainMatcher->featureTracking(prevImage, image, f1Points, f2Points, status))
					return false;
			}
			prevImage = image.clone();
			f1Points.clear();
			f1Points= f2Points;
			f2Points.clear();
			f2Points= f3Points;
		}
	}
	frameNr++;
	return true;
}

void Odometry::fivePoint(const vector<KeyPoint> &xp,
						 const vector<KeyPoint> &x,
						 vector<DMatch> &matches)
{
	inliers.release();
	vector<Point2d> pMTemp, cMTemp;

	pMatchedPoints.clear();
	cMatchedPoints.clear();

	// Copy only matched keypoints
	vector<DMatch>::iterator it;
	for(it = matches.begin(); it != matches.end(); ++it)
	{
		pMTemp.push_back(xp[it->queryIdx].pt);
		cMTemp.push_back(x[it->trainIdx].pt);
	}

	E = findEssentialMat(pMTemp, cMTemp,
						 K, RANSAC, param.odParam.ransacProb,
						 param.odParam.ransacError, inliers);

	// Recover R and t from E
	recoverPose(E, pMTemp, cMTemp, K, R, t, inliers);

	int32_t N = sum(inliers)[0];
	int32_t j = 0;
	pMatchedPoints.resize(N);
	cMatchedPoints.resize(N);
	for(int i = 0; i < (int) inliers.rows; i++)
	{
		if(inliers.at<uint8_t>(i) == 1)
		{
			pMatchedPoints[j] = pMTemp[i];
			cMatchedPoints[j] = cMTemp[i];
			j++;
		}
	}
}

void Odometry::fivePoint(const vector<Point2f> &xp,
						 const vector<Point2f> &x)
{
	inliers.release();

	pMatchedPoints.clear();
	cMatchedPoints.clear();

	E = findEssentialMat(xp, x,
						 K, RANSAC, param.odParam.ransacProb,
						 param.odParam.ransacError, inliers);

	// Recover R and t from E
	recoverPose(E, xp, x, K, R, t, inliers);

	int32_t N = sum(inliers)[0];
	int32_t j = 0;
	pMatchedPoints.resize(N);
	cMatchedPoints.resize(N);
	for(int i = 0; i < (int) inliers.rows; i++)
	{
		if(inliers.at<uint8_t>(i) == 1)
		{
			pMatchedPoints[j] = Point2d(xp[i].x, xp[i].y);
			cMatchedPoints[j] = Point2d( x[i].x,  x[i].y);
			j++;
		}
	}
}

void Odometry::swapAll()
{
	// Swap descriptors
	f1Descriptors = f2Descriptors.clone();
	//f2Descriptors = f3Descriptors.clone();

	// Swap keypoints
	f1Keypoints.clear();
	f1Keypoints = f2Keypoints;
	f2Keypoints.clear();
	//f2Keypoints = f3Keypoints;

	// Swap matches
	//matches12.clear();
	//matches12 = matches23;

	//sharedMatches12.clear();
	//sharedMatches12 = sharedMatches23;

}

bool Odometry::triangulate(const vector<Point2d> &xp,
						   const vector<Point2d> &x,
						   vector<Point3d> &X)
{
	Mat triangPt(4, (uint32_t)x.size(), CV_64FC1);

	Mat pM = Mat::zeros(3, 4, CV_64FC1);
	Mat cM = Mat::zeros(3, 4, CV_64FC1);
	K.copyTo(pM(Range(0, 3), Range(0, 3)));
	R.copyTo(cM(Range(0, 3), Range(0, 3)));
	t.copyTo(cM.col(3));
	cM = K * cM;

	// Triangulate points to 4D
	if(xp.size() < 10)
		return false;
	triangulatePoints(pM, cM, xp, x, triangPt);

	fromHomogeneous(triangPt, X);

	return true;
}

void Odometry::sharedMatches(const vector<DMatch> &m1,
							 const vector<DMatch> &m2,
							 vector<DMatch> &shared1,
							 vector<DMatch> &shared2)
{
	shared1.clear();
	shared2.clear();
	vector<DMatch>::const_iterator it2, it1;
	for(it1 = m1.begin(); it1 != m1.end(); ++it1)
	{
		for(it2 = m2.begin(); it2 != m2.end(); ++it2)
		{
			if(it2->queryIdx == it1->trainIdx)
			{
				shared1.push_back(*it1);
				shared2.push_back(*it2);
				break;
			}
		}
	}
}

bool Odometry::pnp(const vector<Point3d> &X,
				   const vector<Point2d> &x)
{
	inliers.release();
	Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
	rvec = Mat::zeros(3, 1, CV_64FC1); // output rotation vector
	tvec = Mat::zeros(3, 1, CV_64FC1); // output translation vector
	bool useExtrinsicGuess = false;

	solvePnPRansac(X, x, K, distCoeffs, rvec, tvec, useExtrinsicGuess,
				   param.odParam.ransacIterations,
				   param.odParam.ransacError,
				   param.odParam.ransacProb,
				   inliers,
				   param.odParam.pnpFlags);

	Rodrigues(rvec, R); // Convert rotation vector to matrix
	t = tvec;
	return true;
}

void Odometry::sharedFeatures(const vector<KeyPoint> &k1,
							  const vector<KeyPoint> &k2,
							  vector<Point2d> &gk1,
							  vector<Point2d> &gk2,
							  const vector<DMatch> &mask)
{
	gk1.clear();
	gk2.clear();
	vector<DMatch>::const_iterator it;
	for(it = mask.begin(); it != mask.end(); ++it)
	{
		gk1.push_back(k1[it->queryIdx].pt);
		gk2.push_back(k2[it->trainIdx].pt);
	}
}

void Odometry::fromHomogeneous(const Mat &Pt4f, vector<Point3d> &Pt3f)
{
	Pt3f.clear();
	int N = Pt4f.cols; // Number of 4-channel elements
	double x, y, z, w;
	for(int i = 0; i < N; i++)
	{
		// Convert the points to Euclidean space
		w = Pt4f.at<double>(3, i);
		z = Pt4f.at<double>(2, i) / w;
		y = Pt4f.at<double>(1, i) / w;
		x = Pt4f.at<double>(0, i) / w;
		Pt3f.push_back(Point3d(x, y, z));
	}
}

vector<double> Odometry::transformationVec(const Mat &RMat, const Mat &tvec)
{
	double ry = asin( RMat.at<double>(0, 2));
	double rx = asin(-RMat.at<double>(1, 2)) / cos(ry);
	double rz = asin(-RMat.at<double>(0, 1)) / cos(ry);

	vector<double> tr_delta;
	tr_delta.resize(6);
	tr_delta[0] = rx;
	tr_delta[1] = ry;
	tr_delta[2] = rz;
	tr_delta[3] = tvec.at<double>(0);
	tr_delta[4] = tvec.at<double>(1);
	tr_delta[5] = tvec.at<double>(2);
	return tr_delta;
}

Mat Odometry::transformationMat(const vector<double> &tr)
{
	// extract parameters
	double rx = tr[0];
	double ry = tr[1];
	double rz = tr[2];
	double tx = tr[3];
	double ty = tr[4];
	double tz = tr[5];

	// precompute sine/cosine
	double sx = sin(rx);
	double cx = cos(rx);
	double sy = sin(ry);
	double cy = cos(ry);
	double sz = sin(rz);
	double cz = cos(rz);

	// compute transformation
	Mat Tr = Mat::zeros(4, 4, CV_64FC1);
	Tr.at<double>(0, 0) = +cy * cz;
	Tr.at<double>(0, 1) = -cy * sz;
	Tr.at<double>(0, 2) = +sy;
	Tr.at<double>(0, 3) = tx;
	Tr.at<double>(1, 0) = +sx * sy * cz + cx * sz;
	Tr.at<double>(1, 1) = -sx * sy * sz + cx * cz;
	Tr.at<double>(1, 2) = -sx * cy;
	Tr.at<double>(1, 3) = ty;
	Tr.at<double>(2, 0) = -cx * sy * cz + sx * sz;
	Tr.at<double>(2, 1) = +cx * sy * sz + sx * cz;
	Tr.at<double>(2, 2) = +cx * cy;
	Tr.at<double>(2, 3) = tz;
	Tr.at<double>(3, 0) = 0;
	Tr.at<double>(3, 1) = 0;
	Tr.at<double>(3, 2) = 0;
	Tr.at<double>(3, 3) = 1;
	return Tr;
}

void Odometry::computeProjection()
{
}

void Odometry::correctScale(vector<Point3d> &points)
{
	double pitch = param.odParam.pitch;
	double trueHeight = param.odParam.cameraHeight;

	// Compute the scaling factor
	if(gaussKernel(pitch, points, estHeight, param.odParam.motionThreshold))
		rho = trueHeight / estHeight;

	t = t * rho;
}

bool Odometry::getTrueScale(int frame_id)
{

	string line;
	int i = 0;
	ifstream myfile ("/Users/Raman/Documents/Programmering/opencv/VO/odometry/data/poses/00.txt");
	double x = 0, y = 0, z = 0;
	double x_prev, y_prev, z_prev;

	if (myfile.is_open())
	{
		while (( getline (myfile, line) ) && (i <= frame_id))
		{
			z_prev = z;
			x_prev = x;
			y_prev = y;
			std::istringstream in(line);
			//cout << line << '\n';
			for (int j = 0; j < 12; j++)
			{
				in >> z ;
				if (j == 7) y = z;
				if (j == 3)  x = z;
			}

			i++;
		}
		myfile.close();
	}

	else
	{
		cout << "Unable to open file\n";
		return false;
	}
	rho = sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
	t = t * rho;

	return true;
}
