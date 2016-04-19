class vo {
	camera_parameters;

	bool checkKeyframe();
	void triangulate(); // Triangulate points frown 2D to 3D
	void RANSAC();
	void nPointMethod();
	void PnP();
	void rescale();
	void bundle();
	
}
