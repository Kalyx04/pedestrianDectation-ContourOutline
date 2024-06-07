#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    // Create background subtractor
    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2();

    // Open the video file
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file!" << std::endl;
        return -1;
    }

    // Process each frame of the video
    cv::Mat frame;
    while (cap.read(frame)) {
        // Make a copy of the original frame
        cv::Mat frameWithContours = frame.clone();

        // Apply background subtraction to extract foreground
        cv::Mat fgMask;
        bgSubtractor->apply(frame, fgMask);

        // Perform morphological operations to remove noise
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21)));

        // Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw outlines around detected pedestrians
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::drawContours(frameWithContours, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
        }

        // Display the frame with pedestrian outlines
        cv::imshow("Video with Pedestrian Outlines", frameWithContours);

        // Adjust the delay (slow down) - set to 30 milliseconds
        int key = cv::waitKey(30);
        // Exit loop if ESC key is pressed
        if (key == 27) {
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
