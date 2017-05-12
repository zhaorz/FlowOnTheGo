#ifndef __COLOR_H__
#define __COLOR_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void colorFlow(const cv::Mat & inpFlow,
    cv::Mat &rgbFlow,
    const float & max_size,
    bool use_value);

#endif /* __COLOR_H__ */
