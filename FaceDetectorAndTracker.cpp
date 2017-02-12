#include "FaceDetectorAndTracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

FaceDetectorAndTracker::FaceDetectorAndTracker(const std::string cascadeFilePath, const int cameraIndex, size_t numFaces)
{
    m_camera = std::make_unique<cv::VideoCapture>(cameraIndex);
    if (m_camera->isOpened() == false)
    {
        std::cerr << "Failed opening camera" << std::endl;
        exit(-1);
    }

    m_faceCascade = std::make_unique<cv::CascadeClassifier>(cascadeFilePath);
    if (m_faceCascade->empty())
    {
        std::cerr << "Error loading cascade file " << cascadeFilePath << std::endl << 
            "Make sure the file exists" << std::endl;
        exit(-1);
    }

#if CV_VERSION_MAJOR < 3
    m_originalFrameSize.width = (int)m_camera->get(cv::CAP_PROP_FRAME_WIDTH);
    m_originalFrameSize.height = (int)m_camera->get(cv::CAP_PROP_FRAME_HEIGHT);
#else
    m_originalFrameSize.width = (int)m_camera->get(CV_CAP_PROP_FRAME_WIDTH);
    m_originalFrameSize.height = (int)m_camera->get(CV_CAP_PROP_FRAME_HEIGHT);
#endif

    m_downscaledFrameSize.width = m_downscaledFrameWidth;
    m_downscaledFrameSize.height = (m_downscaledFrameSize.width * m_originalFrameSize.height) / m_originalFrameSize.width;

    m_ratio.x = (float)m_originalFrameSize.width / m_downscaledFrameSize.width;
    m_ratio.y = (float)m_originalFrameSize.height / m_downscaledFrameSize.height;

    m_numFaces = numFaces;
}

FaceDetectorAndTracker::~FaceDetectorAndTracker()
{

}

void FaceDetectorAndTracker::operator>>(cv::Mat &frame)
{
    if (m_camera->isOpened() == false)
    {
        frame.release(); 
        return;
    }
    *m_camera >> frame;

    cv::resize(frame, m_downscaledFrame, m_downscaledFrameSize);

    if (!m_tracking) // Search for faces on whole frame until 2 faces are found
    {
        detect();
        return;
    }
    else // if (m_tracking)
    {
        track();
    }
}

std::vector<cv::Rect> FaceDetectorAndTracker::faces()
{
    std::vector<cv::Rect> faces;
    for (const auto& face : m_facesRects)
    {
        faces.push_back(cv::Rect(face.x * m_ratio.x, face.y * m_ratio.y, face.width * m_ratio.x, face.height * m_ratio.y));
    }
    return faces;
}

void FaceDetectorAndTracker::detect()
{
    // Minimum face size is 1/5th of screen height
    // Maximum face size is 2/3rds of screen height
    m_faceCascade->detectMultiScale(m_downscaledFrame, m_facesRects, 1.1, 3, 0,
        cv::Size(m_downscaledFrame.rows / 5, m_downscaledFrame.rows / 5),
        cv::Size(m_downscaledFrame.rows * 2 / 3, m_downscaledFrame.rows * 2 / 3));

    if (m_facesRects.size() < m_numFaces)
    {
        return;
    }
    else if (m_facesRects.size() >= m_numFaces) 
    {
        m_facesRects.resize(m_numFaces);
    }

    // Get face templates
    m_faceTemplates.clear();
    for (auto face : m_facesRects)
    {
        face.width /= 2;
        face.height /= 2;
        face.x += face.width / 2;
        face.y += face.height / 2;
        m_faceTemplates.push_back(m_downscaledFrame(face).clone());
    }

    // Get face ROIs
    m_faceRois.clear();
    for (const auto& face : m_facesRects)
    {
        m_faceRois.push_back(doubleRectSize(face, m_downscaledFrameSize));
    }

    // Initialize template matching timers
    m_tmRunningInRoi.clear();
    m_tmStartTime.clear();
    m_tmEndTime.clear();
    m_tmRunningInRoi.resize(m_facesRects.size(), false);
    m_tmStartTime.resize(m_facesRects.size());
    m_tmEndTime.resize(m_facesRects.size());

    // Turn on tracking
    m_tracking = true;
}

void FaceDetectorAndTracker::track()
{
    for (int i = 0; i < m_faceRois.size(); i++)
    {
        const auto &roi = m_faceRois[i]; // roi

        // Detect faces sized +/-20% off biggest face in previous search
        const cv::Mat &faceRoi = m_downscaledFrame(roi);
        m_faceCascade->detectMultiScale(faceRoi, m_tmpFacesRect, 1.1, 3, 0,
            cv::Size(roi.width * 4 / 10, roi.height * 4 / 10),
            cv::Size(roi.width * 6 / 10, roi.width * 6 / 10));

        if (m_tmpFacesRect.empty()) // No face found in roi... fallback to tm
        {
            if (m_tmStartTime[i] == 0) // if tm just started start stopwatch
            {
                m_tmStartTime[i] = cv::getCPUTickCount();
            }
            
            if (m_faceTemplates[i].cols <= 1 || m_faceTemplates[i].rows <= 1)
            {
                m_facesRects.clear();
                m_tracking = false;
                return;
            }

            // Template matching
            cv::matchTemplate(faceRoi, m_faceTemplates[i], m_matchingResult, CV_TM_SQDIFF_NORMED);
            cv::normalize(m_matchingResult, m_matchingResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            double min, max;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

            // Add roi offset to face position
            m_facesRects[i].x = minLoc.x + roi.x - m_faceTemplates[i].cols / 2;
            m_facesRects[i].y = minLoc.y + roi.y - m_faceTemplates[i].rows / 2;
            m_facesRects[i].width = m_faceTemplates[i].cols * 2;
            m_facesRects[i].height = m_faceTemplates[i].rows * 2;
            
            
            m_tmEndTime[i] = cv::getCPUTickCount();

            double duration = (double)(m_tmEndTime[i] - m_tmStartTime[i]) / cv::getTickFrequency();
            if (duration > m_tmMaxDuration)
            {
                m_facesRects.clear();
                m_tracking = false;
                return; // Stop tracking faces
            }
        }
        else
        {
            m_tmRunningInRoi[i] = false;
            m_tmStartTime[i] = m_tmEndTime[i] = 0;

            m_facesRects[i] = m_tmpFacesRect[0];
            
            m_facesRects[i].x += roi.x;
            m_facesRects[i].y += roi.y;
        }
    }

    for (int i = 0; i < m_facesRects.size(); i++)
    {
        for (int j = i + 1; j < m_facesRects.size(); j++)
        {
            if ((m_facesRects[i] & m_facesRects[j]).area() > 0)
            {
                m_facesRects.clear();
                m_tracking = false;
                return;
            }
        }
    }
}

cv::Rect FaceDetectorAndTracker::doubleRectSize(const cv::Rect &inputRect, const cv::Size &frameSize)
{
    cv::Rect outputRect;
    // Double rect size
    outputRect.width = inputRect.width * 2;
    outputRect.height = inputRect.height * 2;

    // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2;
    outputRect.y = inputRect.y - inputRect.height / 2;

    // Handle edge cases
    if (outputRect.x < 0) {
        outputRect.width += outputRect.x;
        outputRect.x = 0;
    }
    if (outputRect.y < 0) {
        outputRect.height += outputRect.y;
        outputRect.y = 0;
    }

    if (outputRect.x + outputRect.width > frameSize.width) {
        outputRect.width = frameSize.width - outputRect.x;
    }
    if (outputRect.y + outputRect.height > frameSize.height) {
        outputRect.height = frameSize.height - outputRect.y;
    }

    return outputRect;
}
