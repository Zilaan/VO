//Functions used to calculate the relative scale
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include "Scale.h"

using namespace std;

//Function to calculate sigma_h for the skewed Gaussian kernel
double sigma_h(vector<Point3f> points)
{
	vector<double> res;
	if ( points.empty() ) return 0;
	else
	{
		for ( vector<Point3f>::iterator it = points.begin(); it != points.end(); ++it )
		{
			res.push_back( sqrt( pow( it->x, 2  ) + pow ( it->y, 2 ) + pow ( it->z, 2 )) );
		}

		if ( res.size() % 2 == 0 )
			return ( 0.01 * (res[res.size() / 2 - 1] + res[res.size() / 2]) );
		else
			return ( 0.02 * res[res.size() / 2] );
	}
}

//function to estimate the height of the camera
double skew_gauss_kernel(vector<Point3f> xyz)
{
	double sig_h = sigma_h( xyz );
	vector<double> val;
	vector<double>::iterator pos;

	for( vector<Point3f>::iterator it = xyz.begin(); it != xyz.end(); ++it )
	{
		if ( height - it->y > 0)
			val.push_back( exp( (-0.5 * pow( it->y, 2 ) ) / pow( sig_h, 2 ) ) );
		else
			val.push_back( exp( (-0.5 * pow( it->y, 2 ) ) / pow( sig_h * 0.01, 2 ) ) );
	}
	pos = max_element ( val.begin(), val.end() );

	return *pos;
}
