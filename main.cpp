#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main() {

	Mat image;

	namedWindow("Display window");

	VideoCapture cap(0);

	if (!cap.isOpened()) {

		cout << "cannot open camera";

	}

	while (true) {

		cap >> image;


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
		imwrite("contours_retr_list.jpg", image_copy4);
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
	
		//destroyAllWindows();	
	}

	return 0;

}
