#pragma once

#include <Windows.h>

#ifdef HIGH_DIM_LBP_JOINT_BAYESIAN_EXTRACTOR_DLL
#define DLL_API __declspec(dllexport) __stdcall
#else
#define DLL_API __declspec(dllimport) __stdcall
#endif

namespace HighDimLBPJointBayesianExtractor
{

	// initialize extractor
	extern "C" bool DLL_API InitExtractor(const wchar_t *pwzAlignModel, const wchar_t *pJointBayesianModel);


	// face detection
	extern "C" bool DLL_API DetectFaces(__out RECT *prcFacesBuff, int iBuffSize, 
										__out int *piFaceNumInBuff, __out int *piFaceNumInImage,
										const BYTE *pLum, int iWidth, int iHeight);


	// face alignment
	extern "C" bool DLL_API DetectFaceLandmarks(__out float *pfShapeBuff, int iBuffSize, 
											    const BYTE *pLum, int iWidth, int iHeight,
												int iFaceRectX, int iFaceRectY, int iFaceRectWidth, int iFaceRectHeight);

	extern "C" int DLL_API GetLandmarkNumber();


	// extract High-dim LBP feature
	extern "C" bool DLL_API ExtractHighDimLBPFeature(__out BYTE *pHighDimLBPFeaBuff, int iBuffSize, 
													 const BYTE *pLum, int iWidth, int iHeight, 
													 const float *pfShape, int iLandmarkNumber);

	extern "C" int DLL_API GetHighDimLBPFeaLen();


	// joint baysian projection
	extern "C" bool DLL_API ProjectWithJointBayesian(__out float *pHighDimLBPJBFeaBuff, int iBuffSize, 
													 const BYTE *pHighDimLBPFea, int iFeaLen);

	extern "C" int DLL_API GetJointBayesianFeaLen();

	extern "C" float DLL_API CalcFeatureDistance(const float *pfJBFea1, const float *pfJBFea2);


	// release extractor
	extern "C" void DLL_API ReleaseExtractor();

};
