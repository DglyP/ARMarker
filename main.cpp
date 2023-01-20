#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
int thresh = 50;
int N = 5;

void regularDetection(Mat image) {

	cv::Mat img_gray2;
	cv::cvtColor(image, img_gray2, COLOR_BGR2GRAY);
	cv::Mat thresh;
	cv::threshold(img_gray2, thresh, 150, 255, THRESH_BINARY);

	vector<vector<Point>> contours3;
	vector<Vec4i> hierarchy3;
	findContours(thresh, contours3, hierarchy3, RETR_LIST, CHAIN_APPROX_NONE);
	Mat image_copy4 = image.clone();
	drawContours(image_copy4, contours3, -1, Scalar(0, 255, 0), 2);
	imshow("LIST", image_copy4);
	//	waitKey(0);

	vector<vector<Point>> contours4;
	vector<Vec4i> hierarchy4;
	findContours(thresh, contours4, hierarchy4, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	Mat image_copy5 = image.clone();
	drawContours(image_copy5, contours4, -1, Scalar(0, 255, 0), 2);
	imshow("EXTERNAL", image_copy5);
	//waitKey(0);

	vector<vector<Point>> contours5;
	vector<Vec4i> hierarchy5;
	findContours(thresh, contours5, hierarchy5, RETR_CCOMP, CHAIN_APPROX_NONE);
	Mat image_copy6 = image.clone();
	drawContours(image_copy6, contours5, -1, Scalar(0, 255, 0), 2);
	imshow("CCOMP", image_copy6);
	//waitKey(0);

	vector<vector<Point>> contours6;
	vector<Vec4i> hierarchy6;
	findContours(thresh, contours6, hierarchy6, RETR_TREE, CHAIN_APPROX_NONE);
	Mat image_copy7 = image.clone();
	drawContours(image_copy7, contours6, -1, Scalar(0, 255, 0), 2);
	imshow("EXTERNAL", image_copy7);
	//waitKey(0);

	int key = cv::waitKey(25);

	if (key == 'q')
	{
		imshow("Display window", thresh);
	}
	else
	{
		imshow("Display window", image_copy4);
	}

	waitKey(25);
}

void circleCountour(Mat image)
{
	Mat img_gray1;

	cvtColor(image, img_gray1, COLOR_BGR2GRAY);
	Mat thresh1;
	threshold(img_gray1, thresh1, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(thresh1, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_NONE);
	Mat image_copy2 = image.clone();
	drawContours(image_copy2, contours2, -1, Scalar(0, 255, 0), 2);
	imshow("None approximation", image_copy2);
	//waitKey(0);
	Mat image_copy3 = image.clone();
	for (int i = 0; i < contours2.size( ) - 1; i = i + 1) {
		for (int j = 0; j < contours2[i].size() - 1; j = j + 1) {
			circle(image_copy3, (contours2[i][0], contours2[i][1]), 2, Scalar(0, 255, 0), 2);
		}
	}
	imshow("CHAIN_APPROX_SIMPLE Point only", image_copy3);
	waitKey(25);
}

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

static void drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
	for (size_t i = 0; i < squares.size(); i++)
	{

		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		//dont detect the border
		if (p->x > 3 && p->y > 3)
			polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}

	imshow("Detected Squares", image);
}

bool checkIfSquare(vector<Point>& approx)
{
	int x1 = approx[0].x;
	int y1 = approx[0].y;

	int x2 = approx[1].x;
	int y2 = approx[1].y;
	int sum = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
	int perimeter = 0;

	for (int i = 0; i < 4; i++)
	{
		int j = i + 1;
		if (i == 3)
		{
			j = 0;
		}
		perimeter = perimeter + sqrt(pow(approx[j].x - approx[i].x, 2) + pow(approx[j].y - approx[i].y, 2) * 1.0);
	//	cout << perimeter << "\n";
	}

	//cout << "Perimeter is " << perimeter / 4 << " And sum is " << sum << "\n";
	if (perimeter / 4 <= sum - 1 || perimeter / 4 >= sum + 1)
	{
		return true;
	}
	return false;
}

void squareDetector(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

		// down-scale and upscale the image to filter out the noise
		//pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
		//pyrUp(pyr, timg, image.size());


		// blur will enhance edge detection
	Mat image_copy = image.clone();
	Mat timg(image_copy);
	medianBlur(image_copy, timg, 9);
	Mat gray0(timg.size(), CV_8U), gray;

	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 5, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.3) {
						squares.push_back(approx);

						if (checkIfSquare(approx) == false)
						{
							squares.pop_back();
						}
					}
				}
			}
		}
	}
	imshow("Square Check window", image_copy);
	waitKey(25);
}

int main() {
	Mat image;
	VideoCapture cap(0);
	//cap.set(CAP_PROP_FRAME_HEIGHT, 300);
	//cap.set(CAP_PROP_FRAME_WIDTH, 300);

	if (!cap.isOpened()) {
		cout << "cannot open camera";
	}
	while (true) {
		cap >> image;
		//regularDetection(image);
		//circleCountour(image);
		//imshow("Original window", image);
		vector<vector<Point>> squares;
		squareDetector(image, squares);
		drawSquares(image, squares);
		//destroyAllWindows();	
	}
	return 0;

}
