#define _SCL_SECURE_NO_WARNINGS
#include "ATC.h"
#include <chrono>
#include "LandmarkCoreIncludes.h"
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RotationHelpers.h>
#include "GazeEstimation.h"
#include <iostream>
#include <fstream>
ImgData* ImgData::instance=nullptr;
ImgData* ATC::imgDataInstance=nullptr;
//PeopleFeature* PeopleFeature::instance = nullptr;
FeatureHouse* FeatureHouse::instance=nullptr;
ATC* ATC::instance = nullptr;
FeatureHouse* ATC::fhInstance=nullptr;
using std::cout;
using std::endl;

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

//#define EAR_THRESH 0.28
#define EYE_FRAME_MIN 2
#define EYE_FRAME_MAX 8
//timeslice时间片 帧数 一秒30帧
#define TIMESLICE 900
float FeatureHouse::GetDistance(int i, int j)
{
	return sqrt(pow(landmark2D[2 * i] - landmark2D[2 * j], 2) + pow(landmark2D[2 * i + 1] - landmark2D[2 * j + 1], 2));
}

float FeatureHouse::GetDistance3D(int i, int j)
{
	return sqrt(pow(landmark3D[3 * i] - landmark3D[3 * j], 2) + pow(landmark3D[3 * i + 1] - landmark3D[3 * j + 1], 2) + pow(landmark3D[3 * i + 2] - landmark3D[3 * j + 2], 2));
}

float FeatureHouse::EyeAspectRatio(float a, float b, float c) 
{
	return (a + b) / (2 * c);
}

float FeatureHouse::GetEyeDistance(int i, int j)
{
	return sqrt(pow(eye_Landmark2D[2 * i] - eye_Landmark2D[2 * j], 2) + pow(eye_Landmark2D[2 * i + 1] - eye_Landmark2D[2 * j + 1], 2));
}

float FeatureHouse::GetEyeDistance3D(int i, int j)
{
	return sqrt(pow(eye_Landmark3D[3 * i] - eye_Landmark3D[3* j], 2) + pow(eye_Landmark3D[3 * i + 1] - eye_Landmark3D[3 * j + 1], 2) + pow(eye_Landmark3D[3 * i + 2] - eye_Landmark3D[3 * j + 2], 2));
}

cv::Rect FeatureHouse::RectCenterScale(cv::Rect rect, cv::Size size) {
	rect = rect + size;
	cv::Point pt;
	pt.x = cvRound(size.width / 2.0);
	pt.y = cvRound(size.height / 2.0);
	return (rect - pt);
}

FeatureHouse::FeatureHouse() {
	frameNumber = 0;
	effFrameNumber = 0;
	cont_frames = 0;
	cont_frames_mod = 0;
	blink_count = 0;
	threshold = -1;

	svm1 = cv::ml::StatModel::load<cv::ml::SVM>("Eyeoc_svm.xml");
	rtree = cv::ml::StatModel::load<cv::ml::RTrees>("Eyeoc_rtree.xml");

	outFile.open("test.csv", ios::out);
	outFile << "eye_diameter" << ',' << "eye_ratio" << ',';
	outFile << "ear" << ',' << "blink" << ',' << "threshold" << ',' << "maxEar" << ',' << "minEar" << ',';
	//outFile << "res_left" << ',' << "res_right" << ',';
	outFile << "gaze_0_x" << ',' << "gaze_0_y" << ',' << "gaze_0_z" << ',' << "gaze_1_x" << ',' << "gaze_1_y" << ',' << "gaze_1_z" << ',' << " gaze_angle_x" << ',' << " gaze_angle_y" << ',';
	for (int i = 0; i < 56; i++) {
		outFile << " eye_lmk_x_" << i << ',' << "eye_lmk_y_" << i << ',';
	}
	for (int i = 0; i < 56; i++) {
		outFile << " eye_lmk_X_" << i << ',' << "eye_lmk_Y_" << i << ',' << "eye_lmk_Z_" << i << ',';
	}
	for (int i = 0; i < 68; i++) {
		outFile << "x_" << i << ',' << "y_" << i << ',';
	}
	for (int i = 0; i < 68; i++) {
		outFile << "X_" << i << ',' << "Y_" << i << ',' << "Z_" << i << ',';
	}
	outFile << "pose_Tx" << ',' << "pose_Ty" << ',' << "pose_Tz" << ',' << "pose_Rx" << ',' << "pose_Ry" << ',' << "pose_Rz" << ',';
	outFile << endl;
}

bool FeatureHouse::SetFeature(void* face_model, void* parameters,cv::Mat &greyImg, cv::Mat &colorImg, float fx, float fy, float cx, float cy) {
	static cv::Point3f gazeDirection0(0, 0, -1);
	static cv::Point3f gazeDirection1(0, 0, -1);
	cv::Vec2f gaze_angle(0, 0);
	static std::vector<cv::Point2f> eyeLandmark2D;
	static std::vector<cv::Point3f> eyeLandmark3D;
	auto tempFaceModel = reinterpret_cast<LandmarkDetector::CLNF*>(face_model);
	auto tempParameter = reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters);
	confidence = tempFaceModel->detection_certainty;
	//openface calculate
	bool detection_success = LandmarkDetector::DetectLandmarksInVideo(colorImg, *tempFaceModel, *tempParameter, greyImg);
	frameNumber++;
	if (detection_success)
	{
		effFrameNumber++;
		if (tempFaceModel->eye_model)
		{
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection0, fx, fy, cx, cy, true);
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection1, fx, fy, cx, cy, false);
			gaze_angle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
			gaze_angle_x = gaze_angle[0];
			gaze_angle_y = gaze_angle[1];
			eyeLandmark2D = LandmarkDetector::CalculateAllEyeLandmarks(*tempFaceModel);
			eyeLandmark3D = LandmarkDetector::Calculate3DEyeLandmarks(*tempFaceModel, fx, fy, cx, cy);

			for (int i = 0; i < eyeLandmark2D.size(); i++) {
				eye_Landmark2D[2 * i] = eyeLandmark2D[i].x;
				eye_Landmark2D[2 * i + 1] = eyeLandmark2D[i].y;
			}
			for (int i = 0; i < eyeLandmark3D.size(); i++) {
				eye_Landmark3D[3 * i] = eyeLandmark3D[i].x;
				eye_Landmark3D[3 * i + 1] = eyeLandmark3D[i].y;
				eye_Landmark3D[3 * i + 2] = eyeLandmark3D[i].z;
			}
		}
		// Work out the pose of the head from the tracked model
		pose_estimate = LandmarkDetector::GetPose(*tempFaceModel, fx, fy, cx, cy);

		//data copy zone ,use fhInstance->output mutex
		std::lock_guard<std::mutex> lm(output);

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

		//landmark2D
		float tempLandmark[136];
		std::copy(reinterpret_cast<const float*>(tempFaceModel->detected_landmarks.datastart)
			, reinterpret_cast<const float*>(tempFaceModel->detected_landmarks.dataend)
			, tempLandmark);
		//将landmark顺序进行调整，调整成x1 y1 x2 y2...x68 y68
		for (int i = 0; i < 68; i++) {
			landmark2D[2 * i] = tempLandmark[i];
			landmark2D[2 * i + 1] = tempLandmark[i + 68];
		}

		//landmark3D
		cv::Mat1f tempMat = tempFaceModel->GetShape(fx, fy, cx, cy);
		float tempLandmark3D[204];
		std::copy(reinterpret_cast<const float*>(tempMat.datastart)
			, reinterpret_cast<const float*>(tempMat.dataend)
			, tempLandmark3D);
		for (int i = 0; i < 68; i++) {
			landmark3D[3 * i] = tempLandmark3D[i];
			landmark3D[3 * i + 1] = tempLandmark3D[i + 68];
			landmark3D[3 * i + 2] = tempLandmark3D[i + 136];
		}
		
		//求瞳孔变化
		float left_eye_small_diameter = (GetEyeDistance3D(20, 24) + GetEyeDistance3D(21, 25) + GetEyeDistance3D(22, 26) + GetEyeDistance3D(23, 27)) / 4;
		float right_eye_small_diameter = (GetEyeDistance3D(48, 52) + GetEyeDistance3D(49, 53) + GetEyeDistance3D(50, 54) + GetEyeDistance3D(51, 55)) / 4;
		float left_eye_big_diameter = (GetEyeDistance3D(0, 4) + GetEyeDistance3D(1, 5) + GetEyeDistance3D(2, 6) + GetEyeDistance3D(3, 7)) / 4;
		float right_eye_big_diameter = (GetEyeDistance3D(28, 32) + GetEyeDistance3D(29, 33) + GetEyeDistance3D(30, 34) + GetEyeDistance3D(31, 35)) / 4;
		float left_ratio = left_eye_small_diameter / left_eye_big_diameter;
		float right_ratio = right_eye_small_diameter / right_eye_big_diameter;
		//瞳孔直径
		eye_diameter = (left_eye_small_diameter + right_eye_small_diameter) / 2;
		float eye_diameter_big = (left_eye_big_diameter + right_eye_big_diameter) / 2;
		eye_ratio = (left_ratio + right_ratio) / 2;
		//cout << left_eye_big_diameter << " " << left_eye_small_diameter << " " << right_eye_big_diameter << " " << right_eye_small_diameter << endl;
		//cout << eye_diameter << " " << eye_diameter_big << " " << left_ratio << " " << right_ratio << endl;
		

#pragma region SVM
		//用来保存眼部特征点
		std::vector<cv::Point> leftEyeLmk;
		std::vector<cv::Point> rightEyeLmk;

		//特征点保存
		for (int i = 36; i <= 41; i++) {
			cv::Point p(landmark2D[2 * i], landmark2D[2 * i + 1]);
			leftEyeLmk.push_back(p);
		}
		for (int i = 42; i <= 47; i++) {
			cv::Point p(landmark2D[2 * i], landmark2D[2 * i + 1]);
			rightEyeLmk.push_back(p);
		}

		//眼部矩形确定
		cv::Rect temp_left = cv::boundingRect(leftEyeLmk);
		cv::Rect rect_left = RectCenterScale(temp_left, cv::Size(temp_left.height, temp_left.width));
		cv::Rect temp_right = cv::boundingRect(rightEyeLmk);
		cv::Rect rect_right = RectCenterScale(temp_right, cv::Size(temp_right.height, temp_right.width));

		//使用模型
		//左眼
		cv::Mat eye_rect_left = colorImg(rect_left);
		cv::resize(eye_rect_left, eye_rect_left, cv::Size(24, 24));
		cv::Mat eye_gray_left;
		cv::cvtColor(eye_rect_left, eye_gray_left, CV_BGR2GRAY);
	    
		//cv::imshow("left_eye", eye_gray_left);
		eye_gray_left.convertTo(eye_gray_left, CV_32F, 1.0 / 255.0);
		cv::Mat input_eye_left(cv::Size(24 * 24, 1), CV_32F);
		for (int i = 0; i < 24; ++i)
			for (int j = 0; j < 24; ++j)
				input_eye_left.at<float>(i * 24 + j) = eye_gray_left.at<float>(i, j);

		/*float res_left = rtree->predict(input_eye_left);
		cv::Mat tttt;
		rtree->predict(input_eye_left, tttt, cv::ml::StatModel::RAW_OUTPUT);*/

		float res_left = svm1->predict(input_eye_left);

		//右眼
		cv::Mat eye_rect_right = colorImg(rect_right);
		cv::resize(eye_rect_right, eye_rect_right, cv::Size(24, 24));
		cv::Mat eye_gray_right;
		cv::cvtColor(eye_rect_right, eye_gray_right, CV_BGR2GRAY);
		
		//cv::imshow("right_eye", eye_gray_right);
		eye_gray_right.convertTo(eye_gray_right, CV_32F, 1.0 / 255.0);
		cv::Mat input_eye_right(cv::Size(24 * 24, 1), CV_32F);
		for (int i = 0; i < 24; ++i)
			for (int j = 0; j < 24; ++j)
				input_eye_right.at<float>(i * 24 + j) = eye_gray_right.at<float>(i, j);

		/*float res_right = rtree->predict(input_eye_right);
		cv::Mat tttt2;
		rtree->predict(input_eye_right, tttt2, cv::ml::StatModel::RAW_OUTPUT);*/

		float res_right = svm1->predict(input_eye_right);

		cout << res_left << " " << res_right << " ";
		for (int i = 1; i <= 6; i++) {
			cout << headpose3D[i] << " ";
		}
		cout << endl;
#pragma endregion
		

#pragma region EAR
		//左右眼分别计算EAR，再求平均值
		float former_ear = ear;
		float left_eye, right_eye;
		left_eye = EyeAspectRatio(GetDistance3D(37, 41), GetDistance3D(38, 40), GetDistance3D(36, 39));
		right_eye = EyeAspectRatio(GetDistance3D(43, 47), GetDistance3D(44, 46), GetDistance3D(42, 45));
		ear = (left_eye + right_eye) / 2;
		//cout << left_eye << " " << right_eye << endl;
		//cout << GetDistance3D(37, 41) << " " << GetDistance3D(38, 40) << " " << GetDistance3D(36, 39) << endl;

		if (ear > maxEAR)
			maxEAR = ear;
		if (ear < minEAR) 
			minEAR = ear;
		if (ear > tempMaxEAR)
			tempMaxEAR = ear;
		if (ear < tempMinEAR)
			tempMinEAR = ear;

		float former_thresh = threshold;
		threshold = maxEAR - 0.02 > (maxEAR + minEAR) / 2 ? (maxEAR + minEAR) / 2 : maxEAR - 0.02;
		
		//排除睁眼平均ear低于阈值而计数眨眼情况
		//阈值变化大，ear变化小（排除眨眼），前ear低于阈值，现ear高于阈值
		if (former_thresh - threshold >= 0.01 && abs(ear - former_ear) < 0.01 && (former_ear - former_thresh) * (ear - threshold) < 0) {
			cont_frames = 0;
		}

		//刷新阈值
		if (!(effFrameNumber % 10)) {
			maxEAR = tempMaxEAR;
			minEAR = tempMinEAR;
			tempMaxEAR = -1;
			tempMinEAR = 10;
		}
#pragma endregion

		//维护队列，如果眨眼已经过期，则弹出队列
		while (!recentBlink.empty() && ((int)frameNumber - TIMESLICE > recentBlink.front().startFrame)) {
			recentBlink.back().blinkTimeSum -= (recentBlink.front().endFrame - recentBlink.front().startFrame + 1);
			recentBlink.pop();
		}

#pragma region 权重
		////计算眨眼的权重，分别为模型法、ear、总权重
		//int wt_model = 0, wt_ear = 0, wt;

		////计算权重
		////标志位
		//bool sign_mod = false, sign_ear = false;
		////SVM方法
		//if (!res_left && !res_right) {	//两只眼睛都闭
		//	if (startFrame_mod == -1) {
		//		startFrame_mod = frameNumber;
		//	}
		//	cont_frames_mod++;
		//	if (cont_frames_mod >= 2) {
		//		wt_model = 10;
		//	}
		//	wt_model = wt_model > 8 ? wt_model : 8;
		//}
		//else if (!res_left || !res_right) {		//闭一只眼睛
		//	if (startFrame_mod == -1) {
		//		startFrame_mod = frameNumber;
		//	}
		//	wt_model = 3;
		//}
		//else
		//{
		//	if (cont_frames_mod >= 1 && cont_frames_mod < EYE_FRAME_MAX) {
		//		wt_model = 5;
		//	}
		//	//重置
		//	//startFrame_mod = -1;
		//	sign_mod = true;
		//	cont_frames_mod = 0;
		//}

		////EAR方法
		//if (threshold != -1) {
		//	if (ear <= threshold) {
		//		cont_frames++;
		//		if (startFrame_ear == -1) {
		//			startFrame_ear = frameNumber;
		//		}
		//		wt_ear = 5;
		//	}
		//	else {
		//		if (cont_frames >= EYE_FRAME_MIN && cont_frames < EYE_FRAME_MAX) {
		//			wt_ear = 10;
		//		}
		//		//重置
		//		//startFrame_ear = -1;
		//		sign_ear = true;
		//		cont_frames = 0;
		//	}
		//}

		////权重求和
		////权重阈值，一般情况为10，特殊情况时为5（由模型法控制眨眼）。特殊情况包括：EAR初始化阶段
		//int wt_thresh = (threshold == -1) ? 5 : 10;
		//wt = wt_ear + wt_model;
		//if (wt >= wt_thresh) {
		//	if (isBlinking == false) {
		//		if (currentBlink.startFrame == -1) {
		//			currentBlink.startFrame = wt_model > wt_ear ? startFrame_mod : startFrame_ear;
		//			cout << currentBlink.startFrame << " " << startFrame_mod << " " << startFrame_ear << " " << wt_model << " " << wt_ear << endl;
		//		}
		//	}
		//	isBlinking = true;
		//}
		//else
		//{
		//	if (isBlinking == true) {
		//		blink_count++;
		//		currentBlink.endFrame = frameNumber;
		//		currentBlink.blinkTimeSum += (currentBlink.endFrame - currentBlink.startFrame + 1);
		//		if (!recentBlink.empty()) {
		//			currentBlink.blinkTimeSum += recentBlink.back().blinkTimeSum;
		//		}
		//		recentBlink.push(currentBlink);
		//		//cout << currentBlink.startFrame << " " << currentBlink.endFrame << " " << ear<<" "<<res_left<<" " <<res_right  << endl;
		//	}
		//	//重置
		//	currentBlink.startFrame = -1;
		//	currentBlink.blinkTimeSum = 0;
		//	isBlinking = false;
		//}

		//startFrame_mod = (sign_mod) ? -1 : startFrame_mod;
		//startFrame_ear = (sign_ear) ? -1 : startFrame_ear;

#pragma endregion

		//如果EAR低于threshold的次数在某个区间内，就记为1次眨眼
		//同时记录最近10次眨眼的开始帧数、结束帧数，并计算出眨眼时间总和（方便计算）和与上次眨眼的间隔时间
		if (threshold != -1) {
			if (ear <= threshold) {
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
					}
					recentBlink.push(currentBlink);
				}
				//重置
				cont_frames = 0;
				currentBlink.startFrame = -1;
				currentBlink.blinkTimeSum = 0;
			}
		}

		//模型法测试
		//if (!res_left || !res_right) {
		//	cont_frames++;
		//	if (currentBlink.startFrame == -1) {
		//		currentBlink.startFrame = frameNumber;
		//	}
		//}
		//else {
		//	if (cont_frames >= 1 && cont_frames < EYE_FRAME_MAX) {
		//		blink_count++;
		//		currentBlink.endFrame = frameNumber;
		//		currentBlink.blinkTimeSum += (currentBlink.endFrame - currentBlink.startFrame + 1);
		//		if (!recentBlink.empty()) {
		//			currentBlink.blinkTimeSum += recentBlink.back().blinkTimeSum;
		//		}
		//		recentBlink.push(currentBlink);
		//	}
		//	//重置
		//	cont_frames = 0;
		//	currentBlink.startFrame = -1;
		//	currentBlink.blinkTimeSum = 0;
		//}


		//将数据写入csv文件，作为记录
		outFile << eye_diameter << ',' << eye_ratio << ',';
		outFile << ear << ',' << blink_count << ',' << threshold << ',' << maxEAR << ',' << minEAR << ',';
		//outFile << res_left << ',' << res_right << ',';
		for (int i = 0; i < 6; i++) {
			outFile << gazeVector[i] << ',';
		}
		outFile << gaze_angle_x << ',' << gaze_angle_y << ',';
		for (int i = 0; i < 56; i++) {
			outFile << eye_Landmark2D[i * 2] << ',' << eye_Landmark2D[i * 2 + 1] << ',';
		}
		for (int i = 0; i < 56; i++) {
			outFile << eye_Landmark3D[i * 3] << ',' << eye_Landmark3D[i * 3 + 1] << ',' << eye_Landmark3D[i * 3 + 2] << ',';
		}
		for (int i = 0; i < 68; i++) {
			outFile << landmark2D[i * 2] << ',' << landmark2D[i * 2 + 1] << ',';
		}
		for (int i = 0; i < 68; i++) {
			outFile << landmark3D[i * 3] << ',' << landmark3D[i * 3 + 1] << ',' << landmark3D[i * 3 + 2] << ',';
		}
		for (int i = 0; i < 6; i++) {
			outFile << headpose3D[i] << ',';
		}
		outFile << endl;
	}
	else
	{
		//便于显示
		ear = 0;
	}
	if (!recentBlink.empty() && frameNumber % 30 == 0) {
		//要求队列非空且每30帧刷新一次数据
		//计算眨眼频率，队列中眨眼次数 / 总时间，再把帧数换算成时间1800帧=1min，单位：次/min
		blinkFrequency = (float)(recentBlink.size()) * 1800 / (frameNumber > TIMESLICE ? TIMESLICE : frameNumber);
		//计算眨眼间隔，总时间-眨眼消耗的时间 / 队列中眨眼次数，单位：s/次
		blinkInterval = (float)(TIMESLICE - recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//计算眨眼持续时间，眨眼消耗的时间 / 队列中眨眼次数，单位：s/次
		blinkLastTime = (float)(recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//计算perclos，闭眼总时间/总时间*100%
		perclos = (float)(recentBlink.back().blinkTimeSum) / (frameNumber > TIMESLICE ? TIMESLICE : frameNumber) * 100;
	}
	else if (recentBlink.empty()) {
		blinkFrequency = 0;
		blinkInterval = 0;
		blinkLastTime = 0;
		perclos = 0;
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

	/*int open = 0, close = 0;
	string filePath;*/

	Utilities::Visualizer visualizer(true, false, false, false);
	while (threadContinue) {
		//std::cout << "threadContinue "<< std::endl;
		cv::Mat greyImg,colorImg;
		if (imgDataInstance->SetImg()) {
			imgDataInstance->GetGreyImg(greyImg);
			if (useOpenFace) {
				GetColorImg(colorImg);
				detection_success = fhInstance->SetFeature(face_model, parameters, greyImg, colorImg, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
						
				//绘制眼部特征点
				if (detection_success) {
					//保存学习所用数据
					//cv::imshow("eye", eye_gray);
					//if (fhInstance->ear < fhInstance->threshold) {
					//	close++;
					//	filePath = "./data/close/img_" + to_string(close) + ".jpg";
					//}
					//else {
					//	open++;
					//	filePath = "./data/open/img_" + to_string(open) + ".jpg";
					//}
					//cv::imwrite(filePath, eye_gray);

					//绘制眼部矩形
					//cv::rectangle(colorImg, rect, CV_RGB(0, 255, 0));
					//cv::rectangle(colorImg, cv::boundingRect(rightEyeLmk), CV_RGB(0, 255, 0));
					
					//绘制全部特征点
					for (int i = 0; i < 68; i++) {
						cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}


#pragma region paint
					////头部姿态盒子
					//Utilities::DrawBox(colorImg, fhInstance->pose_estimate, cv::Scalar(255, 0, 0), 1.5, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
				
					////绘制视线	
					//float draw_multiplier = 16;
					//int draw_shiftbits = 4;
					////绘制瞳孔的轮廓
					//for (int i = 0; i <= 35; i++) {
					//	if (i == 7) {
					//		cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
					//		cv::Point p2(fhInstance->eye_Landmark2D[2 * 0], fhInstance->eye_Landmark2D[2 * 0 + 1]);
					//		cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
					//		i = 27;
					//		continue;
					//	}
					//	else if (i == 35) {
					//		cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
					//		cv::Point p2(fhInstance->eye_Landmark2D[2 * 28], fhInstance->eye_Landmark2D[2 * 28 + 1]);
					//		cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
					//		break;
					//	}
					//	cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
					//	cv::Point p2(fhInstance->eye_Landmark2D[2 * (i + 1)], fhInstance->eye_Landmark2D[2 * (i + 1) + 1]);
					//	cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
					//}

					//// Now draw the gaze lines themselves
					//cv::Mat cameraMat = (cv::Mat_<float>(3, 3) << imgDataInstance->fx, 0, imgDataInstance->cx, 0, imgDataInstance->fy, imgDataInstance->cy, 0, 0, 0);
					//
					//// Grabbing the pupil location, to draw eye gaze need to know where the pupil is
					//cv::Point3f pupil_left(fhInstance->pupilCenter3D[0], fhInstance->pupilCenter3D[1], fhInstance->pupilCenter3D[2]);
					//cv::Point3f pupil_right(fhInstance->pupilCenter3D[3], fhInstance->pupilCenter3D[4], fhInstance->pupilCenter3D[5]);

					//cv::Point3f gaze_direction0(fhInstance->gazeVector[0], fhInstance->gazeVector[1], fhInstance->gazeVector[2]);
					//cv::Point3f gaze_direction1(fhInstance->gazeVector[3], fhInstance->gazeVector[4], fhInstance->gazeVector[5]);

					//std::vector<cv::Point3f> points_left;
					//points_left.push_back(cv::Point3f(pupil_left));
					//points_left.push_back(cv::Point3f(pupil_left) + cv::Point3f(gaze_direction0)*50.0);

					//std::vector<cv::Point3f> points_right;
					//points_right.push_back(cv::Point3f(pupil_right));
					//points_right.push_back(cv::Point3f(pupil_right) + cv::Point3f(gaze_direction1)*50.0);

					//cv::Mat_<float> proj_points;
					//cv::Mat_<float> mesh_0 = (cv::Mat_<float>(2, 3) << points_left[0].x, points_left[0].y, points_left[0].z, points_left[1].x, points_left[1].y, points_left[1].z);
					//Utilities::Project(proj_points, mesh_0, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
					//cv::line(colorImg, cv::Point(cvRound(proj_points.at<float>(0, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(0, 1) * (float)draw_multiplier)),
					//	cv::Point(cvRound(proj_points.at<float>(1, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(1, 1) * (float)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);

					//cv::Mat_<float> mesh_1 = (cv::Mat_<float>(2, 3) << points_right[0].x, points_right[0].y, points_right[0].z, points_right[1].x, points_right[1].y, points_right[1].z);
					//Utilities::Project(proj_points, mesh_1, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
					//cv::line(colorImg, cv::Point(cvRound(proj_points.at<float>(0, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(0, 1) * (float)draw_multiplier)),
					//	cv::Point(cvRound(proj_points.at<float>(1, 0) * (float)draw_multiplier), cvRound(proj_points.at<float>(1, 1) * (float)draw_multiplier)), cv::Scalar(110, 220, 0), 2, CV_AA, draw_shiftbits);
#pragma endregion

					//瞳孔特征点绘制
					/*for (int i = 48; i <= 55; i++) {
						cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}
					for (int i = 20; i <= 27; i++) {
						cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}*/
				}
				
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
				sprintf(text, "%.2f", fhInstance->perclos);
				string percStr("PERCLOS:");
				percStr += text;
				percStr += "%";

				sprintf(text, "%.4f", fhInstance->eye_diameter);
				string diaStr("EyeDia:");
				diaStr += text;
				sprintf(text, "%.4f", fhInstance->eye_ratio);
				string ratStr("EyeRat:");
				ratStr += text;
				
				/*sprintf(text, "%.2f", fhInstance->gaze_angle_x);
				string gazeXStr("GAZE_X:");
				gazeXStr += text;
				sprintf(text, "%.2f", fhInstance->gaze_angle_y);
				string gazeYStr("GAZE_Y:");
				gazeYStr += text;*/

				cv::putText(colorImg, earStr, cv::Point(20, 40), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, blinkStr, cv::Point(350, 40), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, freStr, cv::Point(20, 90), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, interStr, cv::Point(350, 90), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, lastStr, cv::Point(20, 140), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, percStr, cv::Point(350, 140), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, diaStr, cv::Point(20, 190), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, ratStr, cv::Point(350, 190), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 1, CV_AA);

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
		std::cout << "ERROR: Could not load the landmark detector" << endl;
		return false;
	}
	if (!reinterpret_cast<LandmarkDetector::CLNF*>(face_model)->eye_model)
	{
		std::cout << "WARNING: no eye model found" << endl;
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
	//a->StartThread("2_1.mp4");
	//a->StartThread(0, "test.avi");
	a->StartThread(0);
	
	system("pause");
	return 0;
}