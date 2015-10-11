#include <map>
#include <set>
#include <iostream>
#include <atlenc.h>
#include <atlstr.h>
#include <Windows.h>
#include <gdiplus.h>
#include <stdio.h>
#include "cpprest/json.h"
#include <cpprest/http_listener.h>
#include "cpprest/http_client.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "base64.h"
#include "HighDimLBPJointBayesianExtractor.h"
#include "linearsvm.h"

using namespace Gdiplus;
using namespace HighDimLBPJointBayesianExtractor;

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace web::http::experimental::listener;

using namespace cv;
using namespace std;

#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "HighDimLBPJointBayesianExtractor.lib")


#define TRACE(msg)            wcout << msg
map<utility::string_t, utility::string_t> dictionary;

int feature_number = 57348;
int CLASS_NUMBER;
int n; // Adjusted feature number for SVM
struct model* model_;

HRESULT convertbyteToBYTE(__out BYTE *pLum, char *frame, int w, int h){
	for (int i = 0; i < h * w; i++){
		*pLum = frame[i];
		++pLum;
	}
	return S_OK;
}

string processFrame(Mat image){ 	
	
	
	int w = image.cols;
	int h = image.rows;
	char* frame = new char[w * h];
	BYTE *pLum = new BYTE[w * h];
	frame = (char *)image.data;
	convertbyteToBYTE(pLum, frame, w, h);	

	//face detection
	int iFaceNum = 0;	
	vector<cv::Rect> cvFaces;
	//cvFaces = NativeDetectorWrapper(frame,  w, h);
	iFaceNum = cvFaces.size();


	
	if (iFaceNum == 0){	
		// No face detected
		cout << "no face detected" <<endl;
 		return "0\n";	
	}

	//string response = processFaceRects(iFaceNum, pLum, cvFaces, w, h);
	
	image.release();
	cvFaces.clear();
	delete[] pLum;
	return "";
}


void composeFeatureNode(struct feature_node *x, BYTE* pHighDimLBP, int featureNum, bool useRank){ 
	//compose the feature_node based on our feature analysis
	// if mode == 0, no ranking mode; if mode == 1, ranking mode

	
		int featureNodeCounter;
		for(featureNodeCounter = 0; featureNodeCounter < featureNum ; ++featureNodeCounter){		
			x[featureNodeCounter].index = (featureNodeCounter+1);
			x[featureNodeCounter].value = pHighDimLBP[featureNodeCounter];
		}
		if(model_->bias>=0)
		{
			x[featureNodeCounter].index = n;
			x[featureNodeCounter].value = model_->bias;
			++featureNodeCounter;
		}
		x[featureNodeCounter].index = -1;	

	
}


double doSVMClassification(double target_label, const struct feature_node *x, double* prob_estimates, double& prob){
	double error = 0;

	int nr_class=get_nr_class(model_);
	int n;
	int nr_feature=get_nr_feature(model_);

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int i = 0;
	double predict_label;
	int inst_max_index = 0; // strtol gives 0 if wrong format

	predict_label = predict_probability(model_,x,prob_estimates,prob);	
		
	return predict_label;
}


string processFaceRects(int faceNum, BYTE *pLum, vector<cv::Rect> faces, 
						int img_width, int img_height){
	

	
	for (int i = 0; i < 1; ++i){ 
		cv::Rect face = faces.at(i);

		// Face alignment
		int iLandmarkNum = GetLandmarkNumber();
		float *pfShape = new float [iLandmarkNum*2];
	
		HighDimLBPJointBayesianExtractor::DetectFaceLandmarks(pfShape, iLandmarkNum*2, pLum, img_width, 
			img_height, face.x, face.y, face.width, face.height);
		

		// Feature extraction		
		BYTE *pHighDimLBP = new BYTE[GetHighDimLBPFeaLen()];
		ExtractHighDimLBPFeature(pHighDimLBP, GetHighDimLBPFeaLen(), pLum, img_width, 
							img_height, pfShape, iLandmarkNum);
		
		
		// Classification		
		struct feature_node *x = (struct feature_node *) malloc((feature_number + 2)*sizeof(struct feature_node));
		composeFeatureNode(x, pHighDimLBP, 2124 * 27, true);
	
		int label = 0;	
		double* prob_estimates = (double *) malloc(CLASS_NUMBER*sizeof(double));		
		double prob = 0;
		int prediction = (int)doSVMClassification(double(label), x, prob_estimates, prob);
		// model_->label[i], prob_estimates[i]
		cout << prediction <<endl; 
	}
	
	return "";
}


HRESULT ConvertBitmapToGray(__out BYTE *pLum, Bitmap *pBmp)
{
	BitmapData bmpData;
	if (pBmp->LockBits(NULL, ImageLockModeRead, PixelFormat24bppRGB, &bmpData) != Ok)
		return E_FAIL;
	BYTE *pImageLine = (BYTE*)bmpData.Scan0;
	for (UINT i = 0; i < bmpData.Height; i++)
	{
		BYTE *pImage = pImageLine;
		for (UINT j = 0; j < bmpData.Width; j++)
		{
			BYTE tLuma = BYTE(0.299 * pImage[2] + 0.587 * pImage[1] + 0.114 * pImage[0]);
			*pLum = tLuma;
			pLum++;
			pImage += 3;
		}
		pImageLine += bmpData.Stride;
	}
	pBmp->UnlockBits(&bmpData);
	return S_OK;
}

void handle_post(http_request request)
{
	TRACE("\nhandle POST\n");
	request.reply(status_codes::OK);
	
	web::json::value json_content = request.extract_json().get();
	//wcout << L"Body:" << json_content[L"filename"] << "," << json_content[L"image"]<< endl;
	wcout << L"Body:" << json_content[L"image"].size() << endl;
	
	int width = json_content[L"w"].as_number().to_int32();
	int height = json_content[L"h"].as_number().to_int32();
	

	
	wstring ws = json_content[L"image"].as_string();
	string s(ws.begin(), ws.end());
	//cout << s << endl;
	cout << s.length() << endl;
	//ws.c_str();
	//wcout << ws << endl;
	//wcout << ws.length << endl;
	string decoded_str = base64_decode_tostr(s);
	cout << decoded_str.length() << endl;
	
	vector<char> data(decoded_str.begin(), decoded_str.end());
	//cout << data.size() << endl;
	Mat image(height, width, CV_8UC1, (void *)decoded_str.data());
	
	Mat img = imdecode(image, CV_8UC1);
	cout << img.cols << ", " << img.rows << ", " << img.dims << endl;
	imshow("Image", img);
	//imwrite("test.jpg", img);
	waitKey(10);

	processFrame(img);
	
	
}
 

std::map<std::string,int> ReadSubjs(std::string filePath){
	std::map<std::string,int> subs;
	std::ifstream in;
	in.open(filePath);
	if ( ! in ) {
		printf("Error: Can't open the file named %s.\n", filePath);
		exit(EXIT_FAILURE);
	}
	int index = 0;
	int counter = 0;
	std::string str;
	while ( in ) {  // Continue if the line was sucessfully read.
		getline(in,str); 
		subs.insert(make_pair(str,index));
		++index;
	}
	in.close();
	return subs;
}

void main()
{

	// Init classifier
	string subject_db_path;
	string model_path;
	


	// Init subjectlabel dictionary
	printf("Loading subj dict...\n");
	map<string, int> subjs_dict = ReadSubjs(subject_db_path);

	// Init SVM training model
	printf("Loading svm model...\n");
	if((model_=load_model(model_path.c_str()))==0)
	{
		printf("can't open model file %s\n",model_path);
		exit(EXIT_FAILURE);
	}
	if(!check_probability_model(model_))
	{
		fprintf(stderr, "probability output is only supported for logistic regression\n");
		exit(1);
	}

	// Init Training model
	int nr_feature = get_nr_feature(model_);
	if(model_->bias>=0)
		n = nr_feature + 1;
	else
		n = nr_feature;

	CLASS_NUMBER = get_nr_class(model_);


	// Init Extractor
	if (!InitExtractor(L"E:\\glimpse\\demo\\HighDimLBPJointBayesian\\Model\\Model_Align_100new_27_points.bin", L"E:\\glimpse\\demo\\HighDimLBPJointBayesian\\Model\\Model_JB_SILBP_Ref2000.bin"))
	{
		printf("Init Extractor Failed!\n");
		return;
	}

	http_listener listener(L"http://192.168.5.30:8888");
	listener.support(methods::POST, handle_post);
	
	try {
      listener
         .open()
         .then([&listener](){TRACE(L"\nstarting to listen\n");})
         .wait();
 
      while (true);
   } catch (exception const & e) {

      wcout << e.what() << endl;
   }

	// read Image and convert into gray
	Bitmap bmp(L"Aaron_Eckhart_0001.jpg");
	int iWidth = bmp.GetWidth();
	int iHeight = bmp.GetHeight();
	if (iWidth == 0 || iHeight == 0)
	{
		printf("Read Image Failed!\n");
		return;
	}
	BYTE *pLum = new BYTE[iWidth * iHeight];
	ConvertBitmapToGray(pLum, &bmp);

	// face detection
	const int MAX_FACE_NUM = 16;
	RECT prcFaces[MAX_FACE_NUM];
	int iFaceNum = 0;
	DetectFaces(prcFaces, MAX_FACE_NUM, &iFaceNum, NULL, pLum, iWidth, iHeight);
	printf("Detect %d faces.\n", iFaceNum);
	if (iFaceNum != 1)
		return;

    // face alignment
	RECT rcFace = prcFaces[0];
	int iLandmarkNum = GetLandmarkNumber();
	float *pfShape = new float [iLandmarkNum*2];
	DetectFaceLandmarks(pfShape, iLandmarkNum*2, pLum, iWidth, iHeight, 
		rcFace.left, rcFace.top, rcFace.right-rcFace.left, rcFace.bottom-rcFace.top);

    // high-dim LBP
	BYTE *pHighDimLBP = new BYTE[GetHighDimLBPFeaLen()];
	ExtractHighDimLBPFeature(pHighDimLBP, GetHighDimLBPFeaLen(), pLum, iWidth, iHeight, pfShape, iLandmarkNum);
	
	// joint bayesian
	float *pJBFea = new float[GetJointBayesianFeaLen()];
	ProjectWithJointBayesian(pJBFea, GetJointBayesianFeaLen(), pHighDimLBP, GetHighDimLBPFeaLen());

	// feature distance
	float fDist = CalcFeatureDistance(pJBFea, pJBFea);
	printf("Feature Distance: %f\n", fDist);

	// release model
	ReleaseExtractor();

	delete[] pLum;
	delete[] pfShape;
	delete[] pHighDimLBP;
	delete[] pJBFea;
}
