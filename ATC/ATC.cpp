#define _SCL_SECURE_NO_WARNINGS
#include "ATC.h"
#include <chrono>
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include <iostream>
#include <fstream>
ImgData* ImgData::instance=nullptr;
ImgData* ATC::imgDataInstance=nullptr;
//PeopleFeature* PeopleFeature::instance = nullptr;
FeatureHouse* FeatureHouse::instance=nullptr;
ATC* ATC::instance = nullptr;
FeatureHouse* ATC::fhInstance=nullptr;

#pragma region ImgDataDefine
inline bool ImgData::GetColorImg(cv::Mat &c) {
	if (!isValid)
		return false;
	std::lock_guard<std::mutex> lm(outputMutex);
	c = colorImg.clone();
	return true;
}

inline bool ImgData::GetGreyImg(cv::Mat &g) {
	if (!isValid)
		return false;
	g = greyImg;
	return true;
}

bool ImgData::SetImg() {
	if (!videoCapture.isOpened()) {
		std::cout << "what??" << std::endl;
		throw std::exception("camera | video cann't open!");
	}
	cv::Mat temp;
	if (!videoCapture.read(temp))
	{
		return isValid = false;
	}
	cv::cvtColor(temp, greyImg, CV_BGR2GRAY);
	if (outputFileName != "") {
		colorWriter << temp;
	}
	std::lock_guard<std::mutex> lm(outputMutex);
	colorImg = temp;
	return true;
}

bool ImgData::Open(int index) {
	isValid = videoCapture.open(index);
	SetCameraParam();
	isValid &= videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, width);
	isValid &= videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	isValid &= videoCapture.set(CV_CAP_PROP_FPS, fps);
	outputFileName = "";
	return isValid;
}

bool ImgData::Open(int index, const std::string & fileName) {
	Open(index);
	outputFileName = fileName;
	try {
		isValid=colorWriter.open(fileName, CV_FOURCC('M', 'P', '4', '2'), fps, cv::Size(width, height));
	}
	catch (...) {
		isValid = false;
	}
	return isValid;
}

bool ImgData::Open(const std::string & fileName) {
	outputFileName = "";
	isValid = videoCapture.open(fileName);
	if (isValid) {
		width = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);
		height = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
		fps = videoCapture.get(CV_CAP_PROP_FPS);
		fx = 500.0f*width / 640.0f;
		fy = 500.0f*height / 320.0f;
		cx = width / 2;
		cy = height / 2;
	}
	return isValid;
}
#pragma endregion

#pragma region FeatureHouse

#define EAR_THRESH 0.28
#define EYE_FRAME_MIN 3
#define EYE_FRAME_MAX 8
float FeatureHouse::GetDistance(int i, int j)
{
	return sqrt(pow(landmark2D[2 * (i - 1)] - landmark2D[2 * (j - 1)], 2) + pow(landmark2D[2 * (i - 1) + 1] - landmark2D[2 * (j - 1) + 1], 2));
}

float FeatureHouse::EyeAspectRatio(float a, float b, float c) 
{
	return (a + b) / (2 * c);
}

FeatureHouse::FeatureHouse() {
	frameNumber = 0;
	cont_frames = 0;
	blink_count = 0;

	outFile.open("test.csv", ios::out);
	outFile << "ear" << ',' << "blink" << ',' << "left_eye" << ',' << "right_eye" << ',';
	for (int i = 37; i <= 48; i++) {
		outFile << "x" << i << ',' << "y" << i << ',';
	}
	outFile << endl;
}

bool FeatureHouse::SetFeature(void* face_model, void* parameters,cv::Mat &greyImg, cv::Mat &colorImg, float fx, float fy, float cx, float cy) {
	static cv::Point3f gazeDirection0(0, 0, -1);
	static cv::Point3f gazeDirection1(0, 0, -1);
	cv::Vec2f gaze_angle(0, 0);
	static std::vector<cv::Point3f> eyeLandmark3D;
	auto tempFaceModel = reinterpret_cast<LandmarkDetector::CLNF*>(face_model);
	auto tempParameter = reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters);
	//openface calculate
	bool detection_success = LandmarkDetector::DetectLandmarksInVideo(colorImg, *tempFaceModel, *tempParameter, greyImg);
	ear = 0;
	frameNumber++;
	if (detection_success)
	{
		if (tempFaceModel->eye_model)
		{
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection0, fx, fy, cx, cy, true);
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection1, fx, fy, cx, cy, false);
			gaze_angle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
			gaze_angle_x = gaze_angle[0];
			gaze_angle_y = gaze_angle[1];
			eyeLandmark3D = LandmarkDetector::Calculate3DEyeLandmarks(*tempFaceModel, fx, fy, cx, cy);
		}
		// Work out the pose of the head from the tracked model
		cv::Vec6d pose_estimate = LandmarkDetector::GetPose(*tempFaceModel, fx, fy, cx, cy);

		//data copy zone ,use fhInstance->output mutex
		std::lock_guard<std::mutex> lm(output);


		float tempLandmark[136];
		std::copy(reinterpret_cast<const float*>(tempFaceModel->detected_landmarks.datastart)
			, reinterpret_cast<const float*>(tempFaceModel->detected_landmarks.dataend)
			, tempLandmark);
		//将landmark顺序进行调整，调整成x1 y1 x2 y2...x68 y68
		for (int i = 0; i < 68; i++) {
			landmark2D[2 * i] = tempLandmark[i];
			landmark2D[2 * i + 1] = tempLandmark[i + 68];
		}

		//左右眼分别计算EAR，再求平均值
		float left_eye, right_eye;
		left_eye = EyeAspectRatio(GetDistance(38, 42), GetDistance(39, 41), GetDistance(37, 40));
		right_eye = EyeAspectRatio(GetDistance(44, 48), GetDistance(45, 47), GetDistance(43, 46));
		ear = (left_eye + right_eye) / 2;
		//cout << ear << endl;

		//如果EAR低于EAR_THRESH的次数在某个区间内，就记为1次眨眼
		//同时记录最近10次眨眼的开始帧数、结束帧数，并计算出眨眼时间总和（方便计算）和与上次眨眼的间隔时间
		if (ear <= EAR_THRESH) {
			cont_frames++;
			if (currentBlink.startFrame == -1) {
				currentBlink.startFrame = frameNumber;
			}
		}
		else {
			if (cont_frames >= EYE_FRAME_MIN && cont_frames < EYE_FRAME_MAX) {
				blink_count++;
				currentBlink.endFrame = frameNumber;
				currentBlink.blinkTimeSum += (currentBlink.endFrame - currentBlink.startFrame + 1);
				if (!recentBlink.empty()) {
					currentBlink.blinkTimeSum += recentBlink.back().blinkTimeSum;
					currentBlink.interval = currentBlink.startFrame - recentBlink.back().endFrame - 1;
				}
				else
				{
					currentBlink.interval = currentBlink.startFrame;//初始化第一项的interval，为开始帧的序号
				}
				if (recentBlink.size() >= 10)//如果队列内元素数量大于10，从眨眼时间总和中减去该项并弹出
				{
					currentBlink.blinkTimeSum -= (recentBlink.front().endFrame - recentBlink.front().startFrame + 1);					
					recentBlink.pop();
				}
				recentBlink.push(currentBlink);
			}
			cont_frames = 0;
			currentBlink.startFrame = -1;
			currentBlink.blinkTimeSum = 0;
		}

		//将数据写入csv文件，作为记录
		outFile << ear << ',' << blink_count << ',' << left_eye << ',' << right_eye << ',';
		for (int i = 37; i <= 48; i++) {
			outFile << landmark2D[(i - 1) * 2] << ',' << landmark2D[(i - 1) * 2 + 1] << ',';
		}
		outFile << endl;

		/*float temp;
		for (int i = 1; i < 68; i += 2) {
			temp = landmark2D[i];
			landmark2D[i] = landmark2D[i + 67];
			landmark2D[i + 67] = temp;
		}*/
		for (int i = 0; i < pose_estimate.channels; ++i) {
			headpose3D[i] = pose_estimate[i];
		}
		for (int i = 0; i < 6; ++i) {
			pupilCenter3D[i] = 0.0f;
		}
		//why 8?
		for (int i = 0; i < 8; ++i) {
			//left & right
			pupilCenter3D[0] += eyeLandmark3D[i].x;
			pupilCenter3D[1] += eyeLandmark3D[i].y;
			pupilCenter3D[2] += eyeLandmark3D[i].z;
			pupilCenter3D[3] += eyeLandmark3D[i + eyeLandmark3D.size() / 2].x;
			pupilCenter3D[4] += eyeLandmark3D[i + eyeLandmark3D.size() / 2].y;
			pupilCenter3D[5] += eyeLandmark3D[i + eyeLandmark3D.size() / 2].z;
		}
		for (int i = 0; i < 6; ++i) {
			pupilCenter3D[i] /= 8;
		}
		gazeVector[0] = gazeDirection0.x;
		gazeVector[1] = gazeDirection0.y;
		gazeVector[2] = gazeDirection0.z;
		gazeVector[3] = gazeDirection1.x;
		gazeVector[4] = gazeDirection1.y;
		gazeVector[5] = gazeDirection1.z;
	}
	if (!recentBlink.empty() && frameNumber % 30 == 0) {
		//要求队列非空且每30帧刷新一次数据
		//计算眨眼频率，队列中眨眼次数 / 队列尾-队列头+队头的interval，再把帧数换算成时间1800帧=1min
		blinkFrequency = (float)(recentBlink.size()) * 1800 / (frameNumber - recentBlink.front().startFrame + 1 + recentBlink.front().interval);
		//计算眨眼间隔，队列尾-队列头+队头的interval-眨眼消耗的时间 / 队列中眨眼次数
		blinkInterval = (float)(frameNumber - recentBlink.front().startFrame + 1 + recentBlink.front().interval - recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//计算眨眼持续时间，眨眼消耗的时间 / 队列中眨眼次数
		blinkLastTime = (float)(recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//cout << blinkLastTime << " " << recentBlink.back().blinkTimeSum <<" "<< recentBlink.size()<< endl;
	}
	return detection_success;
}

void FeatureHouse::GetLandmark2d(float landmark2d[68 * 2]) {
	std::lock_guard<std::mutex> lm(output);
	std::copy(landmark2D,landmark2D + 68 * 2, landmark2d);
}

void FeatureHouse::GetPupilCenter3d(float pupilCenter3d[6]) {
	std::lock_guard<std::mutex> lm(output);
	std::copy(pupilCenter3D, pupilCenter3D + 6, pupilCenter3d);
}

void FeatureHouse::GetGazeVector(float gaze[6]) {
	std::lock_guard<std::mutex> lm(output);
	std::copy(gazeVector, gazeVector + 6, gaze);
}

void FeatureHouse::GetHeadPose(float headpose[6]) {
	std::lock_guard<std::mutex> lm(output);
	std::copy(headpose3D, headpose3D + 6, headpose);
}
#pragma endregion

#pragma region ATCDefine
void ATC::ATC_Thread() {
	std::cout << "threadStart" << std::endl;
	cv::VideoWriter writer("test.avi", CV_FOURCC('M', 'P', '4', '2'), 30, cv::Size(imgDataInstance->width, imgDataInstance->height));
	while (threadContinue) {
		//std::cout << "threadContinue "<< std::endl;
		cv::Mat greyImg,colorImg;
		if (imgDataInstance->SetImg()) {
			imgDataInstance->GetGreyImg(greyImg);
			if (useOpenFace) {
				GetColorImg(colorImg);
				detection_success = fhInstance->SetFeature(face_model, parameters, greyImg, colorImg, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
				
				//添加文字
				char text[255];
				sprintf(text, "%.4f", fhInstance->ear);
				string earStr("EAR:");
				earStr += text;
				sprintf(text, "%u", fhInstance->blink_count);
				string blinkStr("BLINK:");
				blinkStr += text;
				sprintf(text, "%.2f", fhInstance->blinkFrequency);
				string freStr("FREQ:");
				freStr += text;
				freStr += "ts/min";
				sprintf(text, "%.2f", fhInstance->blinkInterval);
				string interStr("INTER:");
				interStr += text;
				interStr += "s/ts";
				sprintf(text, "%.2f", fhInstance->blinkLastTime);
				string lastStr("LAST:");
				lastStr += text;
				lastStr += "s/ts";

				sprintf(text, "%.2f", fhInstance->gaze_angle_x);
				string gazeXStr("GAZE_X:");
				gazeXStr += text;
				sprintf(text, "%.2f", fhInstance->gaze_angle_y);
				string gazeYStr("GAZE_Y:");
				gazeYStr += text;
				cv::putText(colorImg, earStr, cv::Point(20, 40), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, blinkStr, cv::Point(350, 40), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, freStr, cv::Point(20, 90), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, interStr, cv::Point(350, 90), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, lastStr, cv::Point(20, 140), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, gazeXStr, cv::Point(20, 190), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, gazeYStr, cv::Point(350, 190), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				
				//绘制眼部特征点
				if (detection_success) {
					for (int i = 36; i <= 47; i++) {
						cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}
				}

				writer << colorImg;
				cv::imshow("test", colorImg);
				cv::waitKey(5);
			}
		}
		else if (!imgDataInstance->IsValid()) {
			break;
		}
	}
	fhInstance->outFile.close();
	std::cout << "thread exit" << std::endl;
}

void ATC::SwitchOpenFace(bool useOpenFace) {
	this->useOpenFace = useOpenFace;
}

cv::Size ATC::StartThread(int index) {
	threadContinue = true;
	if (imgDataInstance->Open(index)) {
		t = new std::thread(std::bind(&ATC::ATC_Thread, this));
		std::cout << imgDataInstance->width << " " << imgDataInstance->height << std::endl;
		return cv::Size(imgDataInstance->width, imgDataInstance->height);
	}
	else {
		return cv::Size(0, 0);
	}
}

cv::Size ATC::StartThread(int index, const std::string & fileName) {
	threadContinue = true;
	if(imgDataInstance->Open(index, fileName)) {
		t = new std::thread(std::bind(&ATC::ATC_Thread, this));
		std::cout << imgDataInstance->width << " " << imgDataInstance->height << std::endl;
		return cv::Size(imgDataInstance->width, imgDataInstance->height);
	}
	else {
		return cv::Size(0, 0);
	}
}

cv::Size ATC::StartThread(const std::string & fileName) {
	threadContinue = true;
	if (imgDataInstance->Open(fileName)) {
		t = new std::thread(std::bind(&ATC::ATC_Thread, this));
		std::cout << imgDataInstance->width << " " << imgDataInstance->height << std::endl;
		return cv::Size(imgDataInstance->width, imgDataInstance->height);
	}
	else {
		return cv::Size(0, 0);
	}
}

void ATC::StopThread() {
	if (t != nullptr) {
		threadContinue = false;
		t->join();
		delete t;
		t = nullptr;
		std::cout << "Stop Thread Over" << std::endl;
		//other resources release
		imgDataInstance->videoCapture.release();
		if (imgDataInstance->outputFileName != "") {
			imgDataInstance->colorWriter.release();
		}
	}
}

bool ATC::GetColorImg(cv::Mat & c)
{
	return threadContinue&&imgDataInstance->GetColorImg(c);
}

bool ATC::GetLandmark2d(float landmark2d[68 * 2]) {
	if (!useOpenFace || !detection_success)
		return false;
	fhInstance->GetLandmark2d(landmark2d);
	return true;
}

bool ATC::GetPupilCenter3d(float pupilCenter3d[6]) {
	if (!useOpenFace || !detection_success)
		return false;
	fhInstance->GetPupilCenter3d(pupilCenter3d);
	return true;
}

bool ATC::GetGazeVector(float gaze[6]) {
	if (!useOpenFace || !detection_success)
		return false;
	fhInstance->GetGazeVector(gaze);
	return true;
}

bool ATC::GetHeadPose(float headpose[6]) {
	if (!useOpenFace || !detection_success)
		return false;
	fhInstance->GetHeadPose(headpose);
	return true;
}

bool ATC::OpenFaceInit(const std::string & exePath) {
	vector<string> arguments = { exePath };
	parameters = new LandmarkDetector::FaceModelParameters(arguments);
	// The modules that are being used for tracking
	std::cout << reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters)->model_location << std::endl;
	face_model=new LandmarkDetector::CLNF(reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters)->model_location);
	if (!reinterpret_cast<LandmarkDetector::CLNF*>(face_model)->loaded_successfully)
	{
		cout << "ERROR: Could not load the landmark detector" << endl;
		return false;
	}
	if (!reinterpret_cast<LandmarkDetector::CLNF*>(face_model)->eye_model)
	{
		cout << "WARNING: no eye model found" << endl;
		return false;
	}
	return true;
}

ATC::~ATC() {
	delete imgDataInstance;
	delete reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters);
	delete reinterpret_cast<LandmarkDetector::CLNF*>(face_model);
	delete fhInstance;
}

#pragma endregion

ofstream outFile;

int main(int argc, char **argv) 
{
	ATC* a = ATC::GetInstance(argv[0], true);
	//a->StartThread("F:\\Project\\ATC\\ATC\\x64\\Release\\YDXJ0004_converter.wmv");
	//a->StartThread("E:\\LYC\\文件\\大学\\学习\\实验室\\陆峰\\人脸识别_空管\\07_12空管实验数据采集\\剪辑_lyc\\管制2摄像头采集\\2_1.mp4");
	//a->StartThread("F:\\FFOutput\\2_1.avi");
	//a->StartThread(0, "test.avi");
	a->StartThread(0);
	
	system("pause");
	return 0;
}