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
	//ͷ����̬��ʼ����־
	bool init_head = false;
	//ͷ����̬�仯��־
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

	//��������ģ��
	void *face_analysis_params;
	void *face_analyser;
	std::vector<std::pair<std::string, double>> au_reg;
	std::vector<std::pair<std::string, double>> au_class;	

	//�洢�����˵Ķ������
	std::vector<int> actions;

	//ͫ��ֱ��
	float eye_diameter;
	//ͫ�׺ͺ�Ĥ�ı�ֵ
	float eye_ratio;

	//SVM������
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
	//���ɹ���֡��
	unsigned int effFrameNumber;
	//ear: the ratio of the eyes' width and height
	float ear = 0;
	//ear��ֵ
	float threshold;
	//blink times
	unsigned int blink_count;
	//number of continuous frames in which ear is less than the THRESH
	unsigned int cont_frames;
	//number of continuous frames in which both of the predictions are 0
	unsigned int cont_frames_mod;
	//ear�����ֵ����Сֵ����ʼ��
	float maxEAR = -1;
	float minEAR = 10;
	//����ˢ��ear�����ֵ����Сֵ
	float tempMaxEAR = -1;
	float tempMinEAR = 10;

	//used to record the recent blink
	typedef struct blink {
		int startFrame = -1;		//գ�ۿ�ʼ֡��
		int endFrame = -1;		//գ�۽���֡��
		unsigned int blinkTimeSum = 0;		//���գ���Լ�֮ǰ��գ�۵ĳ���ʱ��֮�ͣ����ڼ��㣩
		//unsigned int interval = 0;		//���գ�����ϴε�գ�۵ļ��ʱ��
	}blink;
	//���ڼ�¼��ǰգ�۵�״̬���������
	blink currentBlink;
	std::queue<blink> recentBlink;
	float blinkFrequency = 0;
	float blinkInterval = 0;
	float blinkLastTime = 0;

	//��¼���ÿ֡��svm�����true������ۣ����ڼ���perclos�ı���֡��
	std::queue<bool> recentSVM;
	//���۵���ֵ֮�ͣ�����perclos����
	int closeSum = 0;
	//perclos�����۵�֡��/��֡��
	float perclos = 0;

	/*
	//Ȩ�ط���
	//���ַ����Ŀ�ʼ֡��
	int startFrame_mod = -1;
	int startFrame_ear = -1;
	//��־λ
	bool isBlinking = false;*/


#pragma region ע��ɨ�ӿ�
	float gazeLastvector[6] = { 100,100,100,100,100,100 };
	float gaze_last_angle_x = 0;
	float gaze_last_angle_y = 0;
	//ע����֡��
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

	//�Ծ�������Ϊ���ķŴ�size��С
	cv::Rect RectCenterScale(cv::Rect rect, cv::Size size);

	//2֡�����߷���н�
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