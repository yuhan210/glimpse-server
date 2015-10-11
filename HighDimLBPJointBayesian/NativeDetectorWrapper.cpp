#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#using <mscorlib.dll>
#using <Microsoft.FaceSdk.Core.dll>
#using <Microsoft.FaceSdk.Detection.dll>

using namespace Microsoft::FaceSdk::Detection;
using namespace Microsoft::FaceSdk::Image;

std::vector<cv::Rect> NativeDetectorWrapper(char* buf, int width, int height){

	array<unsigned char> ^ img = gcnew array<unsigned char>(width * height);
	for (int i = 0; i < width * height; ++i){
			img[i] = buf[i];
	}
	ImageGray ^grayImage = gcnew ImageGray(img, width, height, width, true);
	FaceDetector ^ detector = gcnew FaceDetector();
	//MultiviewFaceDetector ^ detector = gcnew MultiviewFaceDetector();
	array<FaceRect> ^ faceRects = detector->Detect(grayImage);
	std::vector<cv::Rect> cvFaceRects;
	for (int i = 0; i < faceRects->Length; ++i){
		cv::Rect rect = cv::Rect(faceRects[i].Rect.Left, faceRects[i].Rect.Top, faceRects[i].Rect.Width, faceRects[i].Rect.Height);
		cvFaceRects.push_back(rect);
	}

	return cvFaceRects;
}