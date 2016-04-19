#ifndef ODOMETRY_H
#define ODOMETRY_H

class Odometry
{
public:
	struct parameters
	{
		/*
		 * Some parameters used
		 * for the visual odometry
		 * i.e. focus length etc.
		 */
		parameters ()
		{
			/*
			 * Deafault values for
			 * parameters from above
			 */
		}
	};

	// Constructor, takes as inpute a parameter structure
	Odometry(parameters param);

	// Deconstructor
	virtual ~Odometry();

private:
	/* data */
};

#endif // ODOMETRY_H
