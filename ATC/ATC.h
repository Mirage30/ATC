#pragma once
#include <thread>
#include <mutex>
#include <iostream>
#include <atomic>
#include <opencv2/opencv.hpp>

const unsigned int INIT_WIDTH(1280), INIT_HEIGHT(720);
const float INIT_FX(500.0f), INIT_FY(500.0f), INIT_CX(INIT_WIDTH / 2), INIT_CY(INIT_HEIGHT / 2);
const unsigned int INIT_FPS(30);


//only store color/grey images
class ImgData {
	friend class ATC;
protected:
	static ImgData* instance;
	cv::Mat colorImg;
	cv::Mat greyImg;
	cv::VideoCapture videoCapture;
	std::mutex outputMutex;
	bool isValid;
	//write video to disk
	cv::VideoWriter colorWriter;
	unsigned int width;
	unsigned int height;
	float fx, fy, cx, cy;
	unsigned int fps;
	std::string outputFileName;
	//functions

	static ImgData* GetInstance() {
		if (instance == nullptr) {
			instance = new ImgData();
			instance->SetCameraParam();
		}
		return instance;
	}
	bool SetImg();
	inline bool GetColorImg(cv::Mat &c);
	inline bool GetGreyImg(cv::Mat &g);
	bool Open(int index);
	bool Open(int index, const std::string & fileName);
	bool Open(const std::string & fileName);
	bool IsValid() {
		return isValid;
	}
	//TODO : dynamic change camera parameters
	void SetCameraParam(bool isDefault = true) {
		if (isDefault) {
			width = INIT_WIDTH;
			height = INIT_HEIGHT;
			fx = INIT_FX;
			fy = INIT_FY;
			cx = INIT_CX;
			cy = INIT_CY;
			fps = INIT_FPS;
			outputFileName = "";
		}
	}
	//int GetCameraParam();
};

#pragma region TODO

//store all feature
class FeatureHouse
{
	friend class ATC;
protected:
	static FeatureHouse* instance;
	//x,y,z,euler_x,euler_y,euler_z
	float headpose3D[6];
	//x_left,y_left ,z_left ; x_right,y_right,z_right
	float pupilCenter3D[6];
	//x1,y1,x2,y2.....x68,y68
	float landmark2D[68 * 2];
	//gaze : x,y,z_left ; x,y,z_right 
	float gazeVector[6];

	//lyc
	float ear;
	int blink_count;
	int cont_frames;
	float planeVector[3];
	float planePoint[3];
	float gazePoint[3];
	FeatureHouse() {
		cont_frames = 0;
		blink_count = 0;
		planeVector[0] = 0;
		planeVector[1] = 0;
		planeVector[2] = 1;
		planePoint[0] = -1;
		planePoint[1] = 0;
		planePoint[2] = 0;
	}
	void GazePoint();
	float GetDistance(int i, int j);
	float Eye_aspect_ratio(float a, float b, float c);
	//lyc

	//eye blink count
	unsigned int blink;
	std::mutex output;
	//functions


	//SetFeatures with mutex
	bool SetFeature(void* face_model, void* parameters, cv::Mat &greyImg, cv::Mat &colorImg, float fx, float fy, float cx, float cy);
	static FeatureHouse* GetInstance() {
		if (instance == nullptr) {
			instance = new FeatureHouse();
		}
		return instance;
	}

	//get landmark with mutex
	void GetLandmark2d(float landmark2d[68 * 2]);

	//get pupilCenter with mutex
	void GetPupilCenter3d(float pupilCenter3d[6]);

	//get gaze vector with mutex
	void GetGazeVector(float gaze[6]);

	//get headpose with mutex
	void GetHeadPose(float headpose[6]);
};
#pragma endregion

//ATC for Qt-GUI
class ATC
{
protected:
	ATC() :threadContinue(true), t(nullptr), detection_success(false) {};
	static ATC* instance;
	static ImgData* imgDataInstance;
	static FeatureHouse* fhInstance;
	void ATC_Thread();
	std::thread* t;
	void *parameters;
	void *face_model;
	bool OpenFaceInit(const std::string & exePath);
	std::atomic<bool> threadContinue;
	std::atomic<bool> useOpenFace;
	std::atomic<bool> detection_success;
public:
	~ATC();
	static ATC* GetInstance(const std::string & exePath, bool useOpenFace = false) {
		if (instance == nullptr) {
			instance = new ATC();
			imgDataInstance = ImgData::GetInstance();
			fhInstance = FeatureHouse::GetInstance();
			instance->useOpenFace = useOpenFace;
			instance->parameters = nullptr;
			instance->face_model = nullptr;
			instance->OpenFaceInit(exePath);
		}
		return instance;
	}
	void SwitchOpenFace(bool useOpenFace);

	//Start a thread using webcam(id=index)
	cv::Size StartThread(int index);
	//Start a thread using webcam(id=index), meanwhile save the video to the disk(path=fileName)
	cv::Size StartThread(int index, const std::string & fileName);
	//Start a thread reading video from disk(path=fileName)
	cv::Size StartThread(const std::string & fileName);
	//Stop the current thread, release the relevant resources
	void StopThread();
	//@return false if cv::Mat is invalid ,otherwise return true
	bool GetColorImg(cv::Mat & c);

	//@return false if openface isno't used or didn't work successfully ,otherwise return true
	//@landmark2d: x1,y1 ; x2, y2 ....
	bool GetLandmark2d(float landmark2d[68 * 2]);

	//@return false if openface isno't used or didn't work successfully ,otherwise return true
	//@pupilCenter3d: x_left ,y_left ,z_left ; x_right ,y_right ,z_right
	bool GetPupilCenter3d(float pupilCenter3d[6]);

	//@return false if openface isno't used or didn't work successfully ,otherwise return true
	//@gaze: x_left,y_left,z_left ; x_right ,y_right ,z_right
	bool GetGazeVector(float gaze[6]);

	//@return false if openface isno't used or didn't work successfully ,otherwise return true
	//@headpose: position_x/y/z ; euler_x/y/z
	bool GetHeadPose(float headpose[6]);
};

