#ifndef DETECTORWRAPPER_H
#define DETECTORWRAPPER_H
#using <Microsoft.FaceSdk.Core.dll>
#using <Microsoft.FaceSdk.Detection.dll>

void NativeDetectorWrapper();
std::vector<cv::Rect> NativeDetectorWrapper(char* buf, int width, int height);

#endif