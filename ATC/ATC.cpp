#define _SCL_SECURE_NO_WARNINGS
#include "ATC.h"
#include <chrono>
#include "LandmarkCoreIncludes.h"
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RotationHelpers.h>
#include "FaceAnalyser.h"
#include "GazeEstimation.h"
#include <iostream>
#include <fstream>


//#define EAR_THRESH 0.28
#define EYE_FRAME_MIN 2
#define EYE_FRAME_MAX 8
//timesliceʱ��Ƭ ֡�� һ��30֡
#define TIMESLICE 900

//��ֵ
#define FREQ_THRESH 30
#define INTER_THRESH 5
#define LAST_THRESH 1
#define PERCLOS_THRESH 30

ImgData* ImgData::instance = nullptr;
ImgData* ATC::imgDataInstance = nullptr;
//PeopleFeature* PeopleFeature::instance = nullptr;
FeatureHouse* FeatureHouse::instance = nullptr;
ATC* ATC::instance = nullptr;
FeatureHouse* ATC::fhInstance = nullptr;
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
		isValid = colorWriter.open(fileName, CV_FOURCC('M', 'P', '4', '2'), fps, cv::Size(width, height));
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
	return sqrt(pow(eye_Landmark3D[3 * i] - eye_Landmark3D[3 * j], 2) + pow(eye_Landmark3D[3 * i + 1] - eye_Landmark3D[3 * j + 1], 2) + pow(eye_Landmark3D[3 * i + 2] - eye_Landmark3D[3 * j + 2], 2));
}

cv::Rect FeatureHouse::RectCenterScale(cv::Rect rect, cv::Size size) {
	rect = rect + size;
	cv::Point pt;
	pt.x = cvRound(size.width / 2.0);
	pt.y = cvRound(size.height / 2.0);
	return (rect - pt);
}
float FeatureHouse::GazeCosinDiff(float * gazeLastvector, float * gazeVector)
{
	float a, b, c, d;
	float gazeaverageLastvector[3];
	float averageGaze[3];

	gazeaverageLastvector[0] = gazeLastvector[0];
	gazeaverageLastvector[1] = gazeLastvector[1];
	gazeaverageLastvector[2] = gazeLastvector[2];


	averageGaze[0] = gazeVector[0];
	averageGaze[1] = gazeVector[1];
	averageGaze[2] = gazeVector[2];

	a = sqrt(pow(gazeaverageLastvector[0], 2) + pow(gazeaverageLastvector[1], 2) + pow(gazeaverageLastvector[2], 2));
	b = sqrt(pow(averageGaze[0], 2) + pow(averageGaze[1], 2) + pow(averageGaze[2], 2));
	c = gazeaverageLastvector[0] * averageGaze[0] + gazeaverageLastvector[1] * averageGaze[1] + gazeaverageLastvector[2] * averageGaze[2];
	d = acos(c / (a*b));
	return d;
}
FeatureHouse::FeatureHouse() {
	frameNumber = 0;
	effFrameNumber = 0;
	cont_frames = 0;
	cont_frames_mod = 0;
	blink_count = 0;
	threshold = -1;

	model_L = svm_load_model("Eyeoc_svm_L");
	model_R = svm_load_model("Eyeoc_svm_R");
	svm1 = cv::ml::StatModel::load<cv::ml::SVM>("Eyeoc_svm.xml");
	//rtree = cv::ml::StatModel::load<cv::ml::RTrees>("Eyeoc_rtree.xml");

	outFile.open("test.csv", ios::out);
	outFile << "eye_diameter" << ',' << "eye_ratio" << ',';
	outFile << "ear" << ',' << "blink" << ',' << "threshold" << ',' << "maxEar" << ',' << "minEar" << ',';
	outFile << "b_freq" << ',' << "b_interval" << ',' << "b_last" << ',' << "perclos" << ',';
	//outFile << "res_left" << ',' << "res_right" << ',';
	outFile << "gaze_0_x" << ',' << "gaze_0_y" << ',' << "gaze_0_z" << ',' << "gaze_1_x" << ',' << "gaze_1_y" << ',' << "gaze_1_z" << ',' << " gaze_angle_x" << ',' << " gaze_angle_y" << ',';
	outFile << "gaze_count" << ',' << "gaze_time" << ',' << "saccade_angle_sum" << ',';
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
}

bool FeatureHouse::SetFeature(void* face_model, void* parameters, cv::Mat &greyImg, cv::Mat &colorImg, float fx, float fy, float cx, float cy) {
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

		//face_analyser����
		((FaceAnalysis::FaceAnalyser *)face_analyser)->PredictStaticAUsAndComputeFeatures(colorImg, tempFaceModel->detected_landmarks);
		au_reg = ((FaceAnalysis::FaceAnalyser *)face_analyser)->GetCurrentAUsReg();
		au_class = ((FaceAnalysis::FaceAnalyser *)face_analyser)->GetCurrentAUsClass();
		
		if (!isInit) {
			for (int i = 0; i < au_reg.size(); i++) {
				outFile << au_reg[i].first << "_r" << ',';
			}
			for (int i = 0; i < au_class.size(); i++) {
				outFile << au_class[i].first << "_c" << ',';
			}
			outFile << endl;
			isInit = true;
		}

		//�ж����⶯��
		actions.clear();
		for (int i = 0; i < au_reg.size(); ++i) {
			//cout << au_reg[i].first << " " << au_reg[i].second << endl;
			if (au_reg[i].first == "AU04" && au_reg[i].second >= 1) {
				actions.push_back(4);
			}
			else if (au_reg[i].first == "AU10" && au_reg[i].second >= 0.2) {
				actions.push_back(10);
			}
			else if (au_reg[i].first == "AU12" && au_reg[i].second >= 0.5) {
				actions.push_back(12);
			}
			else if (au_reg[i].first == "AU14" && au_reg[i].second >= 0.5) {
				actions.push_back(14);
			}
			else if (au_reg[i].first == "AU20" && au_reg[i].second >= 1.2) {
				actions.push_back(20);
			}
			else if (au_reg[i].first == "AU26" && au_reg[i].second >= 0.5) {
				actions.push_back(26);
			}
		}
		for (int i = 0; i < au_class.size(); ++i) {
			if (au_class[i].first == "AU25" && au_class[i].second == 1) {
				actions.push_back(25);
			}
		}
		/*for (auto au : actions) {
			cout << au << " ";
		}
		cout << endl;*/


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
		//��landmark˳����е�����������x1 y1 x2 y2...x68 y68
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
#pragma region gazepoint block
		//cout << saccade_angle_sum << endl;
		if (abs(gaze_angle_x - gaze_last_angle_x) < 0.01 || abs(gaze_angle_y - gaze_last_angle_y) < 0.01)
		{
			gaze_frames++;
			saccade_angle_sum = 0;
		}
		else
		{
			saccade_angle_sum += GazeCosinDiff(gazeLastvector, gazeVector);
			if (gaze_frames > 4)
			{
				gaze_count++;
			}
			gaze_frames = 0;
			//gaze_frame_sum += gaze_frames;
		}
		//cout << (abs(gazeaverageLastvector[0] - averageGaze[0]) < 0.05) << endl;
		//cout << "gazeframe" << " " << gaze_frames << endl;
		gaze_last_angle_x = gaze_angle_x;
		gaze_last_angle_y = gaze_angle_y;
		gaze_time = gaze_frames / 25;
		for (int i = 0; i < 5; i++)
		{
			gazeLastvector[i] = gazeVector[i];
		}
		//cout << "gaze_time" << " " << gaze_time << endl;

#pragma endregion		
		//��ͫ�ױ仯
		float left_eye_small_diameter = (GetEyeDistance3D(20, 24) + GetEyeDistance3D(21, 25) + GetEyeDistance3D(22, 26) + GetEyeDistance3D(23, 27)) / 4;
		float right_eye_small_diameter = (GetEyeDistance3D(48, 52) + GetEyeDistance3D(49, 53) + GetEyeDistance3D(50, 54) + GetEyeDistance3D(51, 55)) / 4;
		float left_eye_big_diameter = (GetEyeDistance3D(0, 4) + GetEyeDistance3D(1, 5) + GetEyeDistance3D(2, 6) + GetEyeDistance3D(3, 7)) / 4;
		float right_eye_big_diameter = (GetEyeDistance3D(28, 32) + GetEyeDistance3D(29, 33) + GetEyeDistance3D(30, 34) + GetEyeDistance3D(31, 35)) / 4;
		float left_ratio = left_eye_small_diameter / left_eye_big_diameter;
		float right_ratio = right_eye_small_diameter / right_eye_big_diameter;
		//ͫ��ֱ��
		eye_diameter = (left_eye_small_diameter + right_eye_small_diameter) / 2;
		float eye_diameter_big = (left_eye_big_diameter + right_eye_big_diameter) / 2;
		eye_ratio = (left_ratio + right_ratio) / 2;
		//cout << left_eye_big_diameter << " " << left_eye_small_diameter << " " << right_eye_big_diameter << " " << right_eye_small_diameter << endl;
		//cout << eye_diameter << " " << eye_diameter_big << " " << left_ratio << " " << right_ratio << endl;


#pragma region SVM
		//���������۲�������
		std::vector<cv::Point> leftEyeLmk;
		std::vector<cv::Point> rightEyeLmk;

		//�����㱣��
		for (int i = 36; i <= 41; i++) {
			cv::Point p(landmark2D[2 * i], landmark2D[2 * i + 1]);
			leftEyeLmk.push_back(p);
		}
		for (int i = 42; i <= 47; i++) {
			cv::Point p(landmark2D[2 * i], landmark2D[2 * i + 1]);
			rightEyeLmk.push_back(p);
		}

		//�۲�����ȷ��
		cv::Rect temp_left = cv::boundingRect(leftEyeLmk);
		cv::Rect rect_left = RectCenterScale(temp_left, cv::Size(temp_left.height, temp_left.width));
		cv::Rect temp_right = cv::boundingRect(rightEyeLmk);
		cv::Rect rect_right = RectCenterScale(temp_right, cv::Size(temp_right.height, temp_right.width));
		
		bool left_eye_sign = true;
		bool right_eye_sign = true;
		float res_left = 1;
		float res_right = 1;

		//ʹ��ģ��
		//����
		try {			
			cv::Mat eye_rect_left = colorImg(rect_left);
			cv::resize(eye_rect_left, eye_rect_left, cv::Size(24, 24));
			cv::Mat eye_gray_left;
			cv::cvtColor(eye_rect_left, eye_gray_left, CV_BGR2GRAY);
			//cv::equalizeHist(eye_gray_left, eye_gray_left);
			//cv::imshow("left_eye", eye_gray_left);
			eye_gray_left.convertTo(eye_gray_left, CV_32F, 1.0 / 255.0);
			svm_node* node_left = new svm_node[1 + 576];
			for (int i = 0; i<576; ++i) {
				node_left [i].index = i + 1;
				node_left[i].value = eye_gray_left.at<float>(i / 24, i % 24);
			}
			node_left[576].index = -1;
			double* prob_l = new double[2];
			res_left = (float)svm_predict_probability(model_L, node_left, prob_l);
			delete node_left;
			//cout << res_left << "  " << prob_l[1] << " ";

			/*cv::Mat input_eye_left(cv::Size(24 * 24, 1), CV_32F);
			for (int i = 0; i < 24; ++i)
				for (int j = 0; j < 24; ++j)
					input_eye_left.at<float>(i * 24 + j) = eye_gray_left.at<float>(i, j);
			float res_left = svm1->predict(input_eye_left);*/

			/*float res_left = rtree->predict(input_eye_left);
			cv::Mat tttt;
			rtree->predict(input_eye_left, tttt, cv::ml::StatModel::RAW_OUTPUT);*/		
		}
		catch(...){
			left_eye_sign = false;
		}
		

		//����
		try {
			cv::Mat eye_rect_right = colorImg(rect_right);
			cv::resize(eye_rect_right, eye_rect_right, cv::Size(24, 24));
			cv::Mat eye_gray_right;
			cv::cvtColor(eye_rect_right, eye_gray_right, CV_BGR2GRAY);
			//cv::equalizeHist(eye_gray_right, eye_gray_right);
			//cv::imshow("right_eye", eye_gray_right);
			eye_gray_right.convertTo(eye_gray_right, CV_32F, 1.0 / 255.0);
			svm_node* node_right = new svm_node[1 + 576];
			for (int i = 0; i<576; ++i) {
				node_right[i].index = i + 1;
				node_right[i].value = eye_gray_right.at<float>(i / 24, i % 24);
			}
			node_right[576].index = -1;
			double* prob_r = new double[2];
			res_right = (float)svm_predict_probability(model_R, node_right, prob_r);
			delete node_right;
			//cout << res_right << " " << prob_r[1];

			/*cv::Mat input_eye_right(cv::Size(24 * 24, 1), CV_32F);
			for (int i = 0; i < 24; ++i)
				for (int j = 0; j < 24; ++j)
					input_eye_right.at<float>(i * 24 + j) = eye_gray_right.at<float>(i, j);
			float res_right = svm1->predict(input_eye_right);*/

			/*float res_right = rtree->predict(input_eye_right);
			cv::Mat tttt2;
			rtree->predict(input_eye_right, tttt2, cv::ml::StatModel::RAW_OUTPUT);*/		
		}
		catch (...) {
			right_eye_sign = false;
		}
		
		while (recentSVM.size() > TIMESLICE)
		{
			closeSum = recentSVM.front() ? closeSum - 1 : closeSum;
			recentSVM.pop();
		}

		if (left_eye_sign && right_eye_sign && !res_left && !res_right) {
			recentSVM.push(true);
			closeSum++;
		}
		else
		{
			recentSVM.push(false);
		}

		//cout << endl;
#pragma endregion

		//ͷ��ת��Ƕ�֮��
		float eu_sum = 0;
		headpose_change = false;
		if (init_head) {
			for (int i = 3; i <= 5; i++)
				eu_sum += abs(former_headpose3D[i] - headpose3D[i]);

			//���ͷ��ת������������գ���ж�����
			if (eu_sum > 0.15) {
				cont_frames = 0;
				currentBlink.startFrame = -1;
				currentBlink.blinkTimeSum = 0;
			}

			if (eu_sum > 0.05) {
				headpose_change = true;
				showBox = 15;
			}

			/*
			//ͷ������ı�
			eu_sum = 0;
			for (int i = 0; i <= 2; i++)
				eu_sum += abs(former_headpose3D[i] - headpose3D[i]);*/

			//cout << eu_sum << " ";
			for (int i = 0; i < 6; i++) {
			cout << headpose3D[i] << " ";
			}
			cout << endl;
		}

		std::copy(headpose3D, headpose3D + 6, former_headpose3D);
		init_head = true;

#pragma region EAR
		//�����۷ֱ����EAR������ƽ��ֵ
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

		//ˢ����ֵ
		if (!(effFrameNumber % 10)) {
			maxEAR = tempMaxEAR;
			minEAR = tempMinEAR;
			tempMaxEAR = -1;
			tempMinEAR = 10;
		}

		float former_thresh = threshold;
		threshold = maxEAR - 0.02 > (maxEAR + minEAR) / 2 ? (maxEAR + minEAR) / 2 : maxEAR - 0.02;

		//�ų�����ƽ��ear������ֵ������գ�����
		//��ֵ�仯��ear�仯С���ų�գ�ۣ���ǰear������ֵ����ear������ֵ
		if (former_thresh - threshold >= 0.01 && abs(ear - former_ear) < 0.01 && (former_ear - former_thresh) * (ear - threshold) < 0) {
			cont_frames = 0;
			currentBlink.startFrame = -1;
			currentBlink.blinkTimeSum = 0;
		}

#pragma endregion

		//ά�����У����գ���Ѿ����ڣ��򵯳�����
		while (!recentBlink.empty() && ((int)frameNumber - TIMESLICE > recentBlink.front().startFrame)) {
			recentBlink.back().blinkTimeSum -= (recentBlink.front().endFrame - recentBlink.front().startFrame + 1);
			recentBlink.pop();
		}

#pragma region Ȩ��
		////����գ�۵�Ȩ�أ��ֱ�Ϊģ�ͷ���ear����Ȩ��
		//int wt_model = 0, wt_ear = 0, wt;

		////����Ȩ��
		////��־λ
		//bool sign_mod = false, sign_ear = false;
		////SVM����
		//if (!res_left && !res_right) {	//��ֻ�۾�����
		//	if (startFrame_mod == -1) {
		//		startFrame_mod = frameNumber;
		//	}
		//	cont_frames_mod++;
		//	if (cont_frames_mod >= 2) {
		//		wt_model = 10;
		//	}
		//	wt_model = wt_model > 8 ? wt_model : 8;
		//}
		//else if (!res_left || !res_right) {		//��һֻ�۾�
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
		//	//����
		//	//startFrame_mod = -1;
		//	sign_mod = true;
		//	cont_frames_mod = 0;
		//}

		////EAR����
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
		//		//����
		//		//startFrame_ear = -1;
		//		sign_ear = true;
		//		cont_frames = 0;
		//	}
		//}

		////Ȩ�����
		////Ȩ����ֵ��һ�����Ϊ10���������ʱΪ5����ģ�ͷ�����գ�ۣ����������������EAR��ʼ���׶�
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
		//	//����
		//	currentBlink.startFrame = -1;
		//	currentBlink.blinkTimeSum = 0;
		//	isBlinking = false;
		//}

		//startFrame_mod = (sign_mod) ? -1 : startFrame_mod;
		//startFrame_ear = (sign_ear) ? -1 : startFrame_ear;

#pragma endregion

		//���EAR����threshold�Ĵ�����ĳ�������ڣ��ͼ�Ϊ1��գ��
		//ͬʱ��¼���10��գ�۵Ŀ�ʼ֡��������֡�����������գ��ʱ���ܺͣ�������㣩�����ϴ�գ�۵ļ��ʱ��
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
			//����
			cont_frames = 0;
			currentBlink.startFrame = -1;
			currentBlink.blinkTimeSum = 0;
		}

		//ģ�ͷ�����
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
		//	//����
		//	cont_frames = 0;
		//	currentBlink.startFrame = -1;
		//	currentBlink.blinkTimeSum = 0;
		//}


		//������д��csv�ļ�����Ϊ��¼
		outFile << eye_diameter << ',' << eye_ratio << ',';
		outFile << ear << ',' << blink_count << ',' << threshold << ',' << maxEAR << ',' << minEAR << ',';
		outFile << blinkFrequency << ',' << blinkInterval << ',' << blinkLastTime << ',' << perclos << ',';
		//outFile << res_left << ',' << res_right << ',';
		for (int i = 0; i < 6; i++) {
			outFile << gazeVector[i] << ',';
		}
		outFile << gaze_angle_x << ',' << gaze_angle_y << ',';
		outFile << gaze_count << ',' << gaze_time << ',' << saccade_angle_sum << ',';
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
		for (int i = 0; i < au_reg.size(); i++) {
			outFile << au_reg[i].second << ',';
		}
		for (int i = 0; i < au_class.size(); i++) {
			outFile << au_class[i].second << ',';
		}
		outFile << endl;
	}
	else
	{
		//������ʾ
		ear = 0;
	}
	if (!recentBlink.empty() && frameNumber % 30 == 0) {
		//Ҫ����зǿ���ÿ30֡ˢ��һ������
		//����գ��Ƶ�ʣ�������գ�۴��� / ��ʱ�䣬�ٰ�֡�������ʱ��1800֡=1min����λ����/min
		blinkFrequency = (float)(recentBlink.size()) * 1800 / (frameNumber > TIMESLICE ? TIMESLICE : frameNumber);
		//����գ�ۼ������ʱ��-գ�����ĵ�ʱ�� / ������գ�۴�������λ��s/��
		blinkInterval = (float)(TIMESLICE - recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//����գ�۳���ʱ�䣬գ�����ĵ�ʱ�� / ������գ�۴�������λ��s/��
		blinkLastTime = (float)(recentBlink.back().blinkTimeSum) / (30 * recentBlink.size());
		//����perclos��������ʱ��/��ʱ��*100%
		perclos = (float)(recentBlink.back().blinkTimeSum) / (frameNumber > TIMESLICE ? TIMESLICE : frameNumber) * 100;

		//����perclos��ͨ��SVM�������֡��/��֡��
		float temp = (float)closeSum * 100 / recentSVM.size();
		perclos = temp > perclos ? temp : perclos;
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
	std::copy(landmark2D, landmark2D + 68 * 2, landmark2d);
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
		cv::Mat greyImg, colorImg;
		if (imgDataInstance->SetImg()) {
			imgDataInstance->GetGreyImg(greyImg);
			if (useOpenFace) {
				GetColorImg(colorImg);
				detection_success = fhInstance->SetFeature(face_model, parameters, greyImg, colorImg, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);

				//�����۲�������
				if (detection_success) {
					//�����۲�����
					//cv::rectangle(colorImg, rect, CV_RGB(0, 255, 0));
					//cv::rectangle(colorImg, cv::boundingRect(rightEyeLmk), CV_RGB(0, 255, 0));

					//���ͷ����̬�䶯�������ʾ
					if (fhInstance->showBox > 0) {
						Utilities::DrawBox(colorImg, fhInstance->pose_estimate, cv::Scalar(255, 0, 0), 1.5, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);
						fhInstance->showBox--;
					}

#pragma region paint
					if (GetKeyState(VK_SPACE)) {
						//����ȫ��������
						for (int i = 0; i < 68; i++) {
							cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
							cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
						}					

						////ͷ����̬����
						//Utilities::DrawBox(colorImg, fhInstance->pose_estimate, cv::Scalar(255, 0, 0), 1.5, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);

						//��������	
						float draw_multiplier = 16;
						int draw_shiftbits = 4;
						//����ͫ�׵�����
						for (int i = 0; i <= 35; i++) {
							if (i == 7) {
								cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
								cv::Point p2(fhInstance->eye_Landmark2D[2 * 0], fhInstance->eye_Landmark2D[2 * 0 + 1]);
								cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
								i = 27;
								continue;
							}
							else if (i == 35) {
								cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
								cv::Point p2(fhInstance->eye_Landmark2D[2 * 28], fhInstance->eye_Landmark2D[2 * 28 + 1]);
								cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
								break;
							}
							cv::Point p1(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
							cv::Point p2(fhInstance->eye_Landmark2D[2 * (i + 1)], fhInstance->eye_Landmark2D[2 * (i + 1) + 1]);
							cv::line(colorImg, p1, p2, cv::Scalar(255, 0, 0), 1, CV_AA);
						}

						//// Now draw the gaze lines themselves
						//cv::Mat cameraMat = (cv::Mat_<float>(3, 3) << imgDataInstance->fx, 0, imgDataInstance->cx, 0, imgDataInstance->fy, imgDataInstance->cy, 0, 0, 0);

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
					}
#pragma endregion

					//ͫ�����������
					/*for (int i = 48; i <= 55; i++) {
					cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
					cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}
					for (int i = 20; i <= 27; i++) {
					cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
					cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}*/

					//�۾�����
					/*for (int i = 8; i <= 19; i++) {
						cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}
					for (int i = 36; i <= 47; i++) {
						cv::Point p(fhInstance->eye_Landmark2D[2 * i], fhInstance->eye_Landmark2D[2 * i + 1]);
						cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					}*/
				}

				//�������
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

				sprintf(text, "%u", fhInstance->gaze_count);
				string gazecountStr("GAZECOUNT:");
				gazecountStr += text;
				sprintf(text, "%.2f", fhInstance->gaze_time);
				string gazetimeStr("GAZETIME:");
				gazetimeStr += text;
				sprintf(text, "%.2f", fhInstance->saccade_angle_sum);
				string saccadeanglesumStr("SACCADEAM:");
				saccadeanglesumStr += text;
				//ͷ��λ��
				string headposeStr("HEADPOSE: ");
				string headposeangleStr("HEADPOSEAG: ");
				string leftBrackets("(");
				string rightBrackts(")");
				string comma(",");
				sprintf(text, "%.2f", fhInstance->headpose3D[0]);
				headposeStr += (leftBrackets + text + comma);
				sprintf(text, "%.2f", fhInstance->headpose3D[1]);
				headposeStr += (text + comma);
				sprintf(text, "%.2f", fhInstance->headpose3D[2]);
				headposeStr += (text + rightBrackts);
				//ͷ����ת�Ƕ�
				sprintf(text, "%.2f", fhInstance->headpose3D[3]);
				headposeangleStr += (leftBrackets + text + comma);
				sprintf(text, "%.2f", fhInstance->headpose3D[4]);
				headposeangleStr += (text + comma);
				sprintf(text, "%.2f", fhInstance->headpose3D[5]);
				headposeangleStr += (text + rightBrackts);

				//գ��
				cv::putText(colorImg, blinkStr, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, earStr, cv::Point(20, 40), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				//գ��ͳ��
				if (fhInstance->blinkFrequency < FREQ_THRESH)
					cv::putText(colorImg, freStr, cv::Point(20, 80), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else
					cv::putText(colorImg, freStr, cv::Point(20, 80), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1, CV_AA);

				if (fhInstance->blinkInterval < INTER_THRESH)
					cv::putText(colorImg, interStr, cv::Point(20, 100), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else
					cv::putText(colorImg, interStr, cv::Point(20, 100), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1, CV_AA);

				if (fhInstance->blinkLastTime < LAST_THRESH)
					cv::putText(colorImg, lastStr, cv::Point(20, 120), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else
					cv::putText(colorImg, lastStr, cv::Point(20, 120), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1, CV_AA);

				if (fhInstance->perclos < PERCLOS_THRESH)
					cv::putText(colorImg, percStr, cv::Point(20, 140), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else
					cv::putText(colorImg, percStr, cv::Point(20, 140), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0), 1, CV_AA);
				//ͫ��
				cv::putText(colorImg, diaStr, cv::Point(20, 180), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, ratStr, cv::Point(20, 200), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);

				//���⶯��
				string browStr("brow: ");
				string lipStr("lip: ");
				string jawStr("jaw: ");
				bool brow = false, lip = false, jaw = false;
				cv::Point temp1(fhInstance->landmark2D[2 * 48], fhInstance->landmark2D[2 * 48 + 1]);
				//cv::Point temp2(fhInstance->landmark2D[2 * 54], fhInstance->landmark2D[2 * 54 + 1]);
				for (auto act : fhInstance->actions) {
					switch (act)
					{
					case 4:
						browStr += "brow lowerer";
						brow = true;
						for (int i = 17; i <= 26; i++) {
							cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
							cv::circle(colorImg, p, 2, cv::Scalar(0, 255, 0), -1);
						}
						break;
					/*case 10:
						lipStr = lipStr + (lip ? " & upper lip raiser" : "upper lip raiser");
						lip = true;
						break;*/
					case 12:
						lipStr = lipStr + (lip ? " & lip corner puller" : "lip corner puller");
						lip = true;						
						cv::circle(colorImg, temp1, 2, cv::Scalar(0, 255, 0), -1);
						//cv::circle(colorImg, temp2, 2, cv::Scalar(0, 255, 0), -1);
						for (int i = 54; i <= 59; i++) {
							cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
							cv::circle(colorImg, p, 2, cv::Scalar(0, 255, 0), -1);
						}
						break;
					/*case 14:
						lipStr = lipStr + (lip ? " & dimpler" : "dimpler");
						lip = true;
						break;*/
					/*case 20:
						lipStr = lipStr + (lip ? " & lip strethed" : "lip strethed");
						lip = true;
						break;*/
					case 25:
						lipStr = lipStr + (lip ? " & lip part" : "lip part");
						lip = true;
						for (int i = 60; i <= 67; i++) {
							cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
							cv::circle(colorImg, p, 2, cv::Scalar(0, 255, 0), -1);
						}
						break;
					case 26:
						jawStr += "jaw drop";
						jaw = true;
						for (int i = 5; i <= 11; i++) {
							cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
							cv::circle(colorImg, p, 2, cv::Scalar(0, 255, 0), -1);
						}
						break;
					default:
						break;
					}
				}
				if(brow)
					cv::putText(colorImg, browStr, cv::Point(20, 240), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else {
					browStr += "normal";
					cv::putText(colorImg, browStr, cv::Point(20, 240), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				}

				if (lip)
					cv::putText(colorImg, lipStr, cv::Point(20, 260), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else {
					lipStr += "normal";
					cv::putText(colorImg, lipStr, cv::Point(20, 260), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				}

				if (jaw)
					cv::putText(colorImg, jawStr, cv::Point(20, 280), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				else {
					jawStr += "normal";
					cv::putText(colorImg, jawStr, cv::Point(20, 280), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				}

				//ע��
				cv::putText(colorImg, gazecountStr, cv::Point(450, 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, gazetimeStr, cv::Point(450, 40), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, saccadeanglesumStr, cv::Point(450, 60), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				//AU
				/*for (int i = 0; i < fhInstance->au_class.size(); ++i) {
					sprintf(text, "%.0f", fhInstance->au_class[i].second);
					string auStr(fhInstance->au_class[i].first + ": ");
					auStr += text;
					cv::putText(colorImg, auStr, cv::Point(530, 80 + i * 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				}*/
				for (int i = 0; i < fhInstance->au_reg.size(); ++i) {
					sprintf(text, "%.3f", fhInstance->au_reg[i].second);
					string auStr(fhInstance->au_reg[i].first + ": ");
					auStr += text;
					cv::putText(colorImg, auStr, cv::Point(500, 80 + i * 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				}
				//ͷ����̬
				cv::putText(colorImg, headposeStr, cv::Point(20, 440), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, headposeangleStr, cv::Point(20, 460), CV_FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(255, 0, 0), 1, CV_AA);


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
	if (imgDataInstance->Open(index, fileName)) {
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
	return threadContinue && imgDataInstance->GetColorImg(c);
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
	face_model = new LandmarkDetector::CLNF(reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters)->model_location);

	fhInstance->face_analysis_params = new FaceAnalysis::FaceAnalyserParameters(arguments);
	fhInstance->face_analyser = new FaceAnalysis::FaceAnalyser(*reinterpret_cast<FaceAnalysis::FaceAnalyserParameters*>(fhInstance->face_analysis_params));

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
	/*for (int i = 0; i < argc; i++) {
		cout <<i<<" "<< argv[i] << endl;
	}*/
	ATC* a = ATC::GetInstance(argv[0], true);
	//a->StartThread("F:\\Project\\ATC\\ATC\\x64\\Release\\YDXJ0004_converter.wmv");
	//a->StartThread("E:\\LYC\\�ļ�\\��ѧ\\ѧϰ\\ʵ����\\½��\\����ʶ��_�չ�\\07_12�չ�ʵ�����ݲɼ�\\����_lyc\\����2����ͷ�ɼ�\\2_1.mp4");
	//a->StartThread("2_1.mp4");
	//a->StartThread(0, "test.avi");
	a->StartThread(0);

	system("pause");
	return 0;
}