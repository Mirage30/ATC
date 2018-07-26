#define _SCL_SECURE_NO_WARNINGS
#include "ATC.h"
#include <chrono>
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
ImgData* ImgData::instance = nullptr;
ImgData* ATC::imgDataInstance = nullptr;
//PeopleFeature* PeopleFeature::instance = nullptr;
FeatureHouse* FeatureHouse::instance = nullptr;
ATC* ATC::instance = nullptr;
FeatureHouse* ATC::fhInstance = nullptr;

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

#define EAR_THRESH 0.25
#define EYE_FRAME 3
void FeatureHouse::GazePoint()
{
	float parameTerflag;
	float parameTer;
	parameTerflag = gazeVector[0] * planeVector[0] + gazeVector[1] * planeVector[1] + gazeVector[2] * planeVector[2];
	parameTer = ((planePoint[0] - pupilCenter3D[0])*planeVector[0] + (planePoint[1] - pupilCenter3D[1])*planeVector[1] + (planePoint[2] - pupilCenter3D[2])*planeVector[2]) / parameTerflag;
	gazePoint[0] = pupilCenter3D[0] + gazeVector[0] * parameTer;
	gazePoint[1] = pupilCenter3D[1] + gazeVector[1] * parameTer;
	gazePoint[2] = pupilCenter3D[2] + gazeVector[2] * parameTer;
}
float FeatureHouse::GetDistance(int i, int j)
{
	return sqrt(pow(landmark2D[2 * (i - 1)] - landmark2D[2 * (j - 1)], 2) + pow(landmark2D[2 * (i - 1) + 1] - landmark2D[2 * (j - 1) + 1], 2));
}

float FeatureHouse::Eye_aspect_ratio(float a, float b, float c) {
	return (a + b) / (2 * c);
}


bool FeatureHouse::SetFeature(void* face_model, void* parameters, cv::Mat &greyImg, cv::Mat &colorImg, float fx, float fy, float cx, float cy) {
	static cv::Point3f gazeDirection0(0, 0, -1);
	static cv::Point3f gazeDirection1(0, 0, -1);
	static std::vector<cv::Point3f> eyeLandmark3D;
	auto tempFaceModel = reinterpret_cast<LandmarkDetector::CLNF*>(face_model);
	auto tempParameter = reinterpret_cast<LandmarkDetector::FaceModelParameters*>(parameters);
	//openface calculate
	bool detection_success = LandmarkDetector::DetectLandmarksInVideo(colorImg, *tempFaceModel, *tempParameter, greyImg);
	if (detection_success)
	{
		if (tempFaceModel->eye_model)
		{
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection0, fx, fy, cx, cy, true);
			GazeAnalysis::EstimateGaze(*tempFaceModel, gazeDirection1, fx, fy, cx, cy, false);
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
		for (int i = 0; i < 68; i++) {
			landmark2D[2 * i] = tempLandmark[i];
			landmark2D[2 * i + 1] = tempLandmark[i + 68];
		}

		//lyc
		float left_eye, right_eye;
		left_eye = Eye_aspect_ratio(GetDistance(38, 42), GetDistance(39, 41), GetDistance(37, 40));
		right_eye = Eye_aspect_ratio(GetDistance(44, 48), GetDistance(45, 47), GetDistance(43, 46));
		ear = (left_eye + right_eye) / 2;
		//cout << left_eye << " " << right_eye << endl;
		if (ear <= EAR_THRESH) {
			cont_frames++;
			//cout <<"ear: "<< ear << endl;
		}
		else {
			if (cont_frames >= EYE_FRAME) {
				blink_count++;
				cout << blink_count << endl;
			}
			cont_frames = 0;
		}
		//lyc
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
	GazePoint();

	cout << gazePoint[0] << ' ' << gazePoint[1] << ' ' << gazePoint[2] << endl;
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
	while (threadContinue) {
		//std::cout << "threadContinue "<< std::endl;
		cv::Mat greyImg, colorImg;
		if (imgDataInstance->SetImg()) {
			imgDataInstance->GetGreyImg(greyImg);
			if (useOpenFace) {
				GetColorImg(colorImg);
				detection_success = fhInstance->SetFeature(face_model, parameters, greyImg, colorImg, imgDataInstance->fx, imgDataInstance->fy, imgDataInstance->cx, imgDataInstance->cy);

				char fpsC[255];
				char fpsB[255];
				sprintf(fpsC, "%f", fhInstance->ear);
				string fpsSt("EAR:");
				fpsSt += fpsC;
				sprintf(fpsB, "%d", fhInstance->blink_count);
				string fpsStr("BlINK:");
				fpsStr += fpsB;
				cv::putText(colorImg, fpsSt, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
				cv::putText(colorImg, fpsStr, cv::Point(350, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
				for (int i = 36; i <= 47; i++) {
					cv::Point p(fhInstance->landmark2D[2 * i], fhInstance->landmark2D[2 * i + 1]);
					cv::Point t((imgDataInstance->width / 2 - fhInstance->gazePoint[0]) * 1, (fhInstance->gazePoint[1]) * 1);
					cv::circle(colorImg, p, 2, cv::Scalar(0, 0, 255), -1);
					cv::circle(colorImg, t, 5, cv::Scalar(0, 0, 255), -1);
				}
				cv::imshow("test", colorImg);
				cv::waitKey(5);
			}
		}
		else if (!imgDataInstance->IsValid()) {
			break;
		}
	}
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

int main(int argc, char **argv)
{
	ATC* a = ATC::GetInstance(argv[0], true);
	a->StartThread(0);

	/*cv::VideoCapture captue(0);
	cv::Mat frame;
	while (1) {
	captue >> frame;
	imshow("Œ“µƒ…„œÒÕ∑", frame);
	cv::waitKey(30);
	}*/
	system("pause");
	return 0;
}