#include <opencv2/opencv.hpp>
#include <iostream>


// Функция для создания гистограммы для одного канала и вычисления CDF
std::pair<cv::Mat, cv::Mat> createHistogramAndCDF(const cv::Mat& channel) {
    // Настройки гистограммы
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Вычисление кумулятивной гистограммы (CDF)
    cv::Mat cdf;
    hist.copyTo(cdf);
    for(int i = 1; i < histSize; i++) {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    // Нормализация CDF
    cdf /= cdf.at<float>(histSize - 1);

    // Нормализация и рисование гистограммы
    int hist_w = 256, hist_h = 256;
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
             cv::Scalar(255), 2, 8, 0);
    }
    return {histImage, cdf};
}

// Функция для нахождения пороговых значений alpha1 и alpha2 на основе CDF
void findAlphaThresholds(const cv::Mat& cdf, double& alpha1, double& alpha2, double lower_percent, double upper_percent) {
    // Находим alpha1 и alpha2
    alpha1 = lower_percent * 255;
    alpha2 = upper_percent * 255;
    for (int i = 0; i < 256; i++) {
        if (cdf.at<float>(i) >= lower_percent) {
            alpha1 = i;
            break;
        }
    }
    for (int i = 255; i >= 0; i--) {
        if (cdf.at<float>(i) <= upper_percent) {
            alpha2 = i;
            break;
        }
    }
}

int main() {
    // Загрузка изображения
    cv::Mat srcImage = cv::imread("../christmas.png");
    if (srcImage.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    // Разделение на каналы
    std::vector<cv::Mat> bgr_planes;
    split(srcImage, bgr_planes);

    // Создание гистограмм и CDF для каждого канала
    cv::Mat b_hist, g_hist, r_hist, b_cdf, g_cdf, r_cdf;
    std::tie(b_hist, b_cdf) = createHistogramAndCDF(bgr_planes[0]);
    std::tie(g_hist, g_cdf) = createHistogramAndCDF(bgr_planes[1]);
    std::tie(r_hist, r_cdf) = createHistogramAndCDF(bgr_planes[2]);

    // Вычисление alpha1 и alpha2 для каждого канала
    double alpha1_b, alpha2_b, alpha1_g, alpha2_g, alpha1_r, alpha2_r;
    findAlphaThresholds(b_cdf, alpha1_b, alpha2_b, 0.07, 0.93);
    findAlphaThresholds(g_cdf, alpha1_g, alpha2_g, 0.07, 0.93);
    findAlphaThresholds(r_cdf, alpha1_r, alpha2_r, 0.07, 0.93);

    // Вывод пороговых значений
    std::cout << "Blue channel: alpha1 = " << alpha1_b << ", alpha2 = " << alpha2_b << std::endl;
    std::cout << "Green channel: alpha1 = " << alpha1_g << ", alpha2 = " << alpha2_g << std::endl;
    std::cout << "Red channel: alpha1 = " << alpha1_r << ", alpha2 = " << alpha2_r << std::endl;

    // Создание изображения для совмещения всего
    cv::Mat combinedImage(512, 512, CV_8UC3, cv::Scalar(255,255,255));

    // Помещение исходного изображения в левый верхний угол
    cv::Mat topLeftROI = combinedImage(cv::Rect(0, 0, 256, 256));
    resize(srcImage, topLeftROI, topLeftROI.size(), 0, 0, cv::INTER_LINEAR);

    // Помещение гистограмм в оставшиеся области
    cv::Mat topRightROI = combinedImage(cv::Rect(256, 0, 256, 256));
    cvtColor(r_hist, topRightROI, cv::COLOR_GRAY2BGR);

    cv::Mat bottomLeftROI = combinedImage(cv::Rect(0, 256, 256, 256));
    cvtColor(g_hist, bottomLeftROI, cv::COLOR_GRAY2BGR);

    cv::Mat bottomRightROI = combinedImage(cv::Rect(256, 256, 256, 256));
    cvtColor(b_hist, bottomRightROI, cv::COLOR_GRAY2BGR);

    // Отображение комбинированного изображения
    imwrite("../combined_image_with_histograms.png", combinedImage);

    // Применение автоконтрастирования для каждого канала
    cv::Mat contrasted_b, contrasted_g, contrasted_r;
    bgr_planes[0].convertTo(contrasted_b, CV_8UC1, 255.0 / (alpha2_b - alpha1_b), -alpha1_b * 255.0 / (alpha2_b - alpha1_b));
    bgr_planes[1].convertTo(contrasted_g, CV_8UC1, 255.0 / (alpha2_g - alpha1_g), -alpha1_g * 255.0 / (alpha2_g - alpha1_g));
    bgr_planes[2].convertTo(contrasted_r, CV_8UC1, 255.0 / (alpha2_r - alpha1_r), -alpha1_r * 255.0 / (alpha2_r - alpha1_r));

    // Объединение каналов обратно в одно изображение
    cv::Mat contrasted_image;
    std::vector<cv::Mat> contrasted_planes = {contrasted_b, contrasted_g, contrasted_r};
    merge(contrasted_planes, contrasted_image);

    // Сохранение результата автоконтрастирования
    imwrite("../autcontrasted_image.png", contrasted_image);
    cv::waitKey(0);

    return 0;
}
