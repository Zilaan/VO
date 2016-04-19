Raman class vo {
	camera_parameters;

	Henrik void triangulate(); // Triangulate points frown 2D to 3D
	Raman void RANSAC();
	Henrik void nPointMethod();
	Raman void PnP();

	void rescale();
	bool checkKeyframe();
	void bundle();
	
}
