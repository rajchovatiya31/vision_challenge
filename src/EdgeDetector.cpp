#include<edge_detection/EdgeDetector.hpp>

using namespace edge_detection;
using namespace std;
using namespace cv;
// Your class methods definition goes here

void EdgeDetector::set_image(Mat img)
{	
	/*
	Description: Method to pass image to EdgeDetector class

	img: Image as cv::Mat object to pass 
	*/
	if (img.empty()){
		cout << "Image is null\n";
		exit(0);
	}
	image = img; // assign to img
	output_image = img; // assign to output_image
}

Mat EdgeDetector::image_preprocessing(Mat raw_image, int gaussian_ksize=5, double thresh=0, double max_value=255)
{	
	/*
	Description: This function is aiming to preprocess the image such a way that the raw image will be passed 
				 and using other parameters it will return denoised image. After preprocessing image will be 
				 ready for edge detection.
	image: Raw image pointer.
	gaussian_ksize: Kernel size for gaussian blur filter. Kernel size is intended to be a square size of
					(gaussian_ksize x gaussian_ksize)
	thresh: Threshold used for OTSU thresholding.
	max_value: Maximum value to use with the THRESH_BINARY.
	Return: Preprocessed image. 
	*/
	Mat gray_image, dst /*will hold result*/;
	cvtColor(raw_image, gray_image, COLOR_BGR2GRAY); // Convert color image to gray scale
	GaussianBlur(gray_image, dst, Size(gaussian_ksize, gaussian_ksize), 0); // Blur gray scale image
	threshold(dst, dst, thresh, max_value, CV_THRESH_BINARY | CV_THRESH_OTSU); // Convert gray scale to binay map
	return dst;
}

vector<Vec2f> EdgeDetector::edge_detection(Mat preprocessed_img, int lowThreshold=50, int ratio=4, int kernel_size=3, double rho=1, double theta=CV_PI/180, int threshold=150)
{
	/*
	Description: Function to perform edge detection on the preprocessed_img image.
	preprocessed_img: Preprocessed image from preprocessing step.
	lowThreshold: Lower threshold for hysteresis procedure.
	ratio: Ration to adjust lowThreshold parameter and pass it to threshold2 parameter of Canny.
		   (threshold2= lowThreshold * ratio)
	kernel_size: apertureSize  parameter of Canny.
	rho: Parameter of HoughLines - distance resolution of accumulator in pixel.
	theta: Parameter of HoughLines - angle resolution of accumulator in radians.
	threshold: Parameter of HoughLines - accumulator threshold.
	Return: Vector of lines found by hough transformation. 
	*/

	Mat dst;
	vector<Vec2f> lines; // will hold the results of the detection
	Canny(preprocessed_img, dst, lowThreshold, lowThreshold*ratio, kernel_size); // canny edge detection
    HoughLines(dst, lines, rho, theta, threshold, 0, 0); // hough transformation and find lines
	return lines;
}

void EdgeDetector::draw_lines(Mat img, const vector<Vec2f> &lines, int lineThickness=3)
{
	/*
	Description: Method to draw lines on image found in edge detection step.  
	img: Image on which line has to be drawn.
	lines: Vector of lines found in edge detection step.
	lineThickness: Thickness of line needs to be drawn in pixel. 
	*/
	Point pt1, pt2;
	Scalar lineColor(0,255,0);
	if (lines.size() != 0){
		for(size_t i=0; i <= lines.size(); i++){
			double rho = lines[i][0];  // get line constant
			double theta = lines[i][1]; // get line angle
			double x0 = cos(theta) * rho; // compute x and y
			double y0 = sin(theta) * rho;
			pt1.x = int(x0 + 1000*(-sin(theta))); // compute line ending points
			pt1.y = int(y0 + 1000*(cos(theta)));
			pt2.x = int(x0 - 1000*(-sin(theta)));
			pt2.y = int(y0 - 1000*(cos(theta)));
			line(img, pt1, pt2, lineColor, lineThickness, LINE_AA); // draw single line
		}
	}
}

void EdgeDetector::show_image(Mat img, string windowName="Output")
{
	/*
	Description: Method to display the image.  
	img: Image to be displayed.
	windowName: Name of the display window
	*/
	if (!(img.empty())){
		namedWindow(windowName, WINDOW_AUTOSIZE); // set window name and size
		imshow(windowName, img); // disply image
		waitKey(); // wait for key to press
	}
	else{
		cout << "Invalid image to show\n";
	}
}

/* This methods are yet to be implement. This methods are for croping the excessive lines from rotated chess
image. 

map<int, vector<Vec2f> > EdgeDetector::segmented_by_angle_kmeans(vector<Vec2f> lines, int k=2)
{
	// Discription: To segment hough lines based on their angles.

	vector<Vec2f> angles;
	vector<Point2f> centers;
	vector<int> labels;
	for (int i = 0; i < lines.size(); i++) {
        float angle = lines[i][1];
        angles.push_back(angle);
    }

	Mat points(angles.size(), 2, CV_32F);
    for (int j = 0; j < angles.size(); j++) {
        float angle = 2 * angles[j][1];
        points.at<float>(j, 0) = cos(angle);
        points.at<float>(j, 1) = sin(angle);
    }
	TermCriteria term(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0);
	kmeans(points, k, labels, term, 10, KMEANS_RANDOM_CENTERS, centers);
	// labels.reshape(-1);
	cout << labels.size() << '\n';
	map<int, vector<Vec2f> > segmented;
	for (size_t i = 0; i < lines.size(); ++i)
    {
        segmented[labels[i]].push_back(lines[i]);
    }

	// vector<Vec2f> result;
	// typedef map <int, vector<Vec2f>> MapType;
	// for (MapType::iterator it = segmented.begin(); it != segmented.end(); ++it){
	// 	result.push_back(it->second);
	// }
    return segmented;
	
}

vector<int> EdgeDetector::intersection(Vec2f line1, Vec2f line2)
{
	// Discription: Find intersection of two lines.

	double rho1 = line1[0]; double theta1 = line1[1];
	double rho2 = line2[0]; double theta2 = line2[1];

	double a1 = 1 / atan(theta1); double a2 = 1 / atan(theta2);
	int x = int(rho2 - rho1 / (a1 - a2));
	int y = int((a1 * rho2 - a2 * rho1) / (a1 - a2));
	vector<int> result(x,y);
	return result;
}

vector<vector<int> > EdgeDetector::find_intersections(vector<Vec2f> lines)
{
	// Discription: Step to find intersections of all hough lines.

	vector<vector<int> > intersections;
	for(size_t i=0; i<= (lines.size() - 1); i++){
		for(size_t j=lines.size(); j <= i+1; j--){
			intersections.push_back(EdgeDetector::intersection(lines[i], lines[j]));
		}
	}
}
*/

void EdgeDetector::detect_edges()
{
	/*
	Description: Main method that combine all steps of process. 
	*/
	Mat preprocessed_img;
	vector<Vec2f> lines;
	map<int, vector<Vec2f> > intersections;
	if (!(image.empty())){
		preprocessed_img = EdgeDetector::image_preprocessing(image); // preprocess step
		lines = EdgeDetector::edge_detection(preprocessed_img);  // edge detection step
		EdgeDetector::draw_lines(output_image, lines); // draw lines on image
		EdgeDetector::show_image(output_image);  // display the image
	}
	else{
		cout << "The image is empty in edge detection.\n";
	}

}

int main(int argc, char** argv)
{
	edge_detection::EdgeDetector detector;
    // Create the executable for testing the code here
	
	string default_file = "../data/Image_1.png";
	string filename = argc == 2 ? argv[1] : default_file;

	Mat image;
	image = imread(filename);

	detector.set_image(image);
	detector.detect_edges();
	return 0;
}

