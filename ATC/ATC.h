#pragma once
#include <thread>
#include <mutex>
#include <iostream>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "svm.h"
#include <fstream>
#include <queue>

const unsigned int INIT_WIDTH(640), INIT_HEIGHT(480);
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
	double confidence;
	//x,y,z,euler_x,euler_y,euler_z
	float headpose3D[6];
	float former_headpose3D[6];
	//头部姿态初始化标志
	bool init_head = false;
	//头部姿态变化标志
	bool headpose_change = false;
	int showBox;
	//headpose
	cv::Vec6d pose_estimate;
	//x_left,y_left ,z_left ; x_right,y_right,z_right
	float pupilCenter3D[6];
	//x1,y1,x2,y2.....x68,y68
	float landmark2D[68 * 2];
	//X1,Y1,Z1,X2...X68,Y68,Z68
	float landmark3D[68 * 3];
	//gaze : x,y,z_left ; x,y,z_right 
	float gazeVector[6];
	//gaze_angle_x,y
	float gaze_angle_x;
	float gaze_angle_y;
	//eyelandmark
	float eye_Landmark2D[56 * 2];
	float eye_Landmark3D[56 * 3];

	//脸部分析模型
	void *face_analysis_params;
	void *face_analyser;
	std::vector<std::pair<std::string, double>> au_reg;
	std::vector<std::pair<std::string, double>> au_class;	

	//存储发生了的动作序号
	std::vector<int> actions;

	//瞳孔直径
	float eye_diameter;
	//瞳孔和虹膜的比值
	float eye_ratio;

	//SVM分类器
	cv::Ptr<cv::ml::SVM> svm1;
	svm_model* model_L;
	svm_model* model_R;

	//Random Forest
	//cv::Ptr<cv::ml::RTrees> rtree;

	//csv writer
	std::ofstream outFile;
	//csv initial sign
	bool isInit = false;

	//frame number
	unsigned int frameNumber;
	//检测成功的帧数
	unsigned int effFrameNumber;
	//ear: the ratio of the eyes' width and height
	float ear = 0;
	//ear阈值
	float threshold;
	//blink times
	unsigned int blink_count;
	//number of continuous frames in which ear is less than the THRESH
	unsigned int cont_frames;
	//number of continuous frames in which both of the predictions are 0
	unsigned int cont_frames_mod;
	//ear的最大值和最小值，初始化
	float maxEAR = -1;
	float minEAR = 10;
	//用于刷新ear的最大值和最小值
	float tempMaxEAR = -1;
	float tempMinEAR = 10;

	//used to record the recent blink
	typedef struct blink {
		int startFrame = -1;		//眨眼开始帧号
		int endFrame = -1;		//眨眼结束帧号
		unsigned int blinkTimeSum = 0;		//这次眨眼以及之前的眨眼的持续时间之和（便于计算）
		//unsigned int interval = 0;		//这次眨眼与上次的眨眼的间隔时间
	}blink;
	//用于记录当前眨眼的状态，方便入队
	blink currentBlink;
	std::queue<blink> recentBlink;
	float blinkFrequency = 0;
	float blinkInterval = 0;
	float blinkLastTime = 0;

	//记录最近每帧的svm结果，true代表闭眼，用于计算perclos的闭眼帧数
	std::queue<bool> recentSVM;
	//闭眼的数值之和，用于perclos计算
	int closeSum = 0;
	//perclos：闭眼的帧数/总帧数
	float perclos = 0;

	/*
	//权重方法
	//两种方法的开始帧号
	int startFrame_mod = -1;
	int startFrame_ear = -1;
	//标志位
	bool isBlinking = false;*/


#pragma region 注视扫视块
	float gazeLastvector[6] = { 100,100,100,100,100,100 };
	float gaze_last_angle_x = 0;
	float gaze_last_angle_y = 0;
	//注视总帧数
	float gaze_frame_sum;
	float gaze_time;
	//gaze event
	unsigned int gaze_count;
	////number of continuous frames in gaze occurance
	unsigned int gaze_frames;
	//used to record the recent gaze
	typedef struct gaze {
		int startFrame = -1;
		int endFrame = -1;
		unsigned int gazeTimesum = 0;
		unsigned int interval = 0;
	}gaze;
	gaze currentGaze;
	float saccade_angle_sum;
#pragma endregion

	std::mutex output;
	//functions

	FeatureHouse();
	//calculate the distance of two landmarks in 2D
	float GetDistance(int i, int j);

	//calculate the distance of two landmarks in 3D
	float GetDistance3D(int i, int j);

	//calculate the ear
	float EyeAspectRatio(float a, float b, float c);

	//calculate the distance of two eyelandmarks in 2D
	float GetEyeDistance(int i, int j);

	//calculate the distance of two eyelandmarks in 3D
	float GetEyeDistance3D(int i, int j);

	//以矩形中心为中心放大size大小
	cv::Rect RectCenterScale(cv::Rect rect, cv::Size size);

	//2帧间视线方向夹角
	float GazeCosinDiff(float *gazeaverageLastvector, float *averageGaze);

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