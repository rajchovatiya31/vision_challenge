#pragma once

#include <opencv2/opencv.hpp>
#include<string>
#include<vector>
#include<math.h>

using namespace std;
using namespace cv;
namespace edge_detection
{

class EdgeDetector
{
	// Your class declaration goes here 
private:
	Mat image;
	Mat output_image;
	
	Mat image_preprocessing(Mat raw_image, int gaussian_ksize, double thresh, double max_value); // Preprocess image
	vector<Vec2f> edge_detection(Mat preprocessed_img, int lowThreshold, int ratio, int kernel_size, double rho, double theta, int threshold); // detect edges
	map<int, vector<Vec2f> > segmented_by_angle_kmeans(vector<Vec2f> lines, int k); // Segment lines
	vector<vector<int> > find_intersections(vector<Vec2f> lines); // Helping method to find intersection of two line
	vector<int> intersection(Vec2f line1, Vec2f line2); // Find intesections of all hough lines

public:
	EdgeDetector(Mat image)
		:image(image), output_image(image.clone()){ };  //Parameterised constructor
	EdgeDetector(){ };  // Default constructor
	~EdgeDetector(){ };

	void set_image(Mat image);  // Helping method to set image when default constructor is called
	void draw_lines(Mat output_image, const vector<Vec2f> &lines, int lineThickness); //Helping method to draw lines on image
	void show_image(Mat output_image, string windowName); // Helping method to display image
	void detect_edges(); // Main method to combine whole process
	
};

}