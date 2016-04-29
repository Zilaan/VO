//Functions used to calculate the relative scale
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

float sigma_h(vector<T> q)//Change T to proper type!*!*!*!*!*!*!*!*!*!
{
	vector<T> res;
	if ( q.empty() ) return 0;
	else	{
		for ( vector<T>::iterator it = q.begin(); it != q.end(); ++it )
		{
			res.push_back( sqrt( pow( it.pt.x, 2  ) + pow ( it.pt.y, 2 ) + pow ( it.pt.z, 2 )) );//not sure if pt.y
		}
		
		if ( res.size() % 2 == 0 )
			return ( 0.01 * (res[res.size()/2 - 1] + res[res.size()/2]) );
		else
			return ( 0.01 * res[res.size()] );
	}
}

float skew_gauss_kernel(float height, vector<T> xyz)
{
	float sig_h = sigma_h( xyz );
	vector<float> val;
	vector<float>::iterator pos;	

	for( vector<T>::iterator it = xyz.begin(); it != xyz.end(); ++it )
	{
		if ( height - it.pt.y > 0)
			val.push_back( exp( (-0.5*pow( it.pt.y, 2 ) )/pow( sig_h, 2 ) ) );
		else
			val.push_back( exp( (-0.5*pow( it.pt.y,2 ) )/pow( sig_h*0.01, 2 ) ) );
	}
	pos = max_element ( val.begin(), val.end() );
	
	return *pos;
} 

