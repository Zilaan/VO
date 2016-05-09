//Functions used to calculate the relative scale
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

float sigma_h(vector<Point3f> q)//Function to calculate sigma_h for the skewed Gaussian kernel
{
        vector<float> res;
        if ( q.empty() ) return 0;
        else    {
                for ( vector<Point3f>::iterator it = q.begin(); it != q.end(); ++it )
                {
                        res.push_back( sqrt( pow( it->x, 2 ) + pow ( it->y, 2 ) + pow ( it->z, 2 ) ) );
                }

                sort( res.begin(), res.end());

                if ( res.size() % 2 == 0 )
                        return ( 0.01 * (res[res.size()/2 - 1] + res[res.size()/2]) );
                else
                        return ( 0.02 * res[res.size()/2] );
        }
}
 float gaussKernel(float pitch, vector<Point3f> xyz)//function to estimate the height of the camera
{
        float h, sig_h = sigma_h( xyz );
        cout << "SIGMA_H: " << sig_h << endl;
        vector<float> val;
        vector<float>::iterator pos;
        for( vector<Point3f>::iterator it2 = xyz.begin(); it2 != xyz.end(); ++it2 )
        {
                h = it2->y * cos(pitch) - it2->z * sin(pitch);
                //cout << "Calculated h: " << h << endl;
                for( vector<Point3f>::iterator it = xyz.begin(); it != xyz.end(); ++it )
                {
                        if ( it != it2 ){
                                if ( h - it->y > 0)
                                        val.push_back( exp( (-0.5*pow( h-it->y, 2 ) )/pow( sig_h, 2 ) ) );
                                else
                                        val.push_back( exp( (-0.5*pow( h-it->y,2 ) )/pow( sig_h*0.01, 2 ) ) );
                        }
                }
        }
        for (vector<float>::iterator it3 = val.begin(); it3 != val.end() ; ++it3) {
                //cout << *it3 << endl;
        }
        pos = max_element ( val.begin(), val.end() );
        cout << "HEIGHT: " << *pos << endl;
        return *pos;
}
