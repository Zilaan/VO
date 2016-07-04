#include "Odometry.h"
#include <iostream>
#include <ctime>
#include <fstream>
#include <cstdio>
#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace cv;

// 00 4541
// 01 1101
// 02 4661
// 03 801
// 04 271
const int lastFrame = 1034;

void writeMatrix(vector< vector<double> > &res)
{
	FILE *myfile = fopen("/Users/Raman/Documents/Programmering/opencv/VO/odometry/results/data/00.txt", "w");

	vector< vector<double> >::iterator it;

	for(it = res.begin(); it < res.end(); it++)
	{
		vector<double> temp = *it;
		for(int i = 0; i < 11; i++)
			fprintf(myfile, "%e ", temp[i]);
			//myfile << temp[i] << " ";

		fprintf(myfile, "%e\n", temp[11]);
	}

	fclose(myfile);
}

void writeToFile(double *p)
{
	ofstream myfile;
	myfile.open("data.csv");
	int idx = 0;
	for(int i = 0; i < lastFrame - 1; i++)
	{
		myfile << *(p + idx) << "," << *(p + idx + 1) << "\n";
		idx = idx + 2;
	}

	myfile.close();
}

int main(int argc, const char * argv[])
{
	char imageName[100];
	int frame;
	Mat prev_Tr = Mat::eye(4, 4, CV_64FC1);
	Mat new_Tr, Tr, invTr, temp;
	double x, y;

	double *ptr = new double[2 * lastFrame];
	vector< vector<double> > result;
	int idx = 0;

	Odometry::parameters param;

	// Extractors Descriptors Matcher
	// 0: Fast    0: BRIEF    0: BF
	// 1: ORB     1: ORB      1: FLANN: KDTree
	// 2: SURF    2: SURF     2: FLANN: LSH
	// 3: SIFT    3: SIFT
	param.maParam.extractor        = 0;
	param.maParam.descriptor       = 1;
	param.maParam.matcher          = 2;
	param.maParam.bucketing        = 0;
	param.odParam.f                = 718.856;
	param.odParam.cu               = 607.1928;
	param.odParam.cv               = 185.2157;
	param.odParam.cameraHeight     = 1.6;
	param.odParam.pitch            = -0.08;
	param.odParam.pnpFlags         = cv::SOLVEPNP_P3P;
	param.odParam.ransacIterations = 1000; // Only PNP
	param.odParam.ransacError      = 0.5;
	param.odParam.ransacProb       = 0.999;
	param.odParam.scaling          = 2;
	param.odParam.motionThreshold  = 100;
	param.odParam.method           = 0;
	param.odParam.doBundle         = 0;
	param.odParam.bundleParam      = 20;
	param.odParam.imageSequence    = 10;

	DIR* dir = opendir("/Users/Raman/Documents/Programmering/opencv/VO/odometry/results/data/");
	if (dir)
	{
		/* Directory exists. */
		closedir(dir);
	}
	else if (ENOENT == errno)
	{
		/* Directory does not exist. */
		cout << "Directory /Users/Raman/Documents/Programmering/opencv/VO/odometry/results/data/ does not exist." << endl;
		return -1;
	}
	else
	{
		/* opendir() failed for some other reason. */
		cout << "Directory ./results/data/ does not exist" << endl;
		return -1;
	}

	Odometry *viso = new Odometry(param);

	clock_t start = clock();
	for (frame = 0; frame < lastFrame; frame++)
	{
		//sprintf(imageName,
		//		"/users/raman/documents/programmering/chalmers/exjobb/libviso2/img/leftdata/%06d.png", frame);
		switch (param.odParam.imageSequence)
		{
			case 0:
				sprintf(imageName,
						"/Volumes/RamanExtern/Data/dataset/sequences/00/image_0/%06d.png", frame);
				break;
			case 1:
				sprintf(imageName,
						"/Volumes/RamanExtern/Data/dataset/sequences/01/image_0/%06d.png", frame);
				break;
			case 2:
				sprintf(imageName,
						"/Volumes/RamanExtern/Data/dataset/sequences/02/image_0/%06d.png", frame);
				break;
			case 3:
				sprintf(imageName,
						"/Volumes/RamanExtern/Data/dataset/sequences/03/image_0/%06d.png", frame);
				break;
			case 4:
				sprintf(imageName,
						"/Volumes/RamanExtern/Data/dataset/sequences/04/image_0/%06d.png", frame);
				break;
			default:
				sprintf(imageName,
						"/Users/Raman/Documents/Programmering/Chalmers/Exjobb/libviso2/img/LeftData/%06d.png", frame);
		}

		Mat image = imread(imageName, IMREAD_GRAYSCALE);
		bool ok = viso->process(image);
		if (ok && frame > 0)
		{
			Tr = viso->getMotion();
			invTr = Tr.inv();
			temp = prev_Tr * invTr;
			new_Tr = temp.clone();
		}
		else
		{
			new_Tr = prev_Tr.clone();
		}

		if(frame > 0)
		{
			x = prev_Tr.at<double>(0, 3);
			y = prev_Tr.at<double>(2, 3);
			*(ptr + idx) = x;
			*(ptr + idx + 1) = y;
			idx = idx + 2;
			cout << "I: " << frame << " x: " << x << " y: " << y << endl;
		}
		prev_Tr = new_Tr.clone();

		vector<double> temp;
		temp.assign((double*)prev_Tr.datastart, (double*)prev_Tr.dataend);
		result.push_back(temp);
	}
	clock_t end = clock();
	cout << "Total time: " << double(end - start) / CLOCKS_PER_SEC << " sec" << endl;;
	cout << "Time/frame: " << double(end - start) / CLOCKS_PER_SEC / lastFrame << " sec/frame" << endl;;

	writeToFile(ptr);
	writeMatrix(result);
	cout << "Data saved" << endl;

	delete viso;
	delete[] ptr;
	cout << "Closed\n";
	return 0;
}
