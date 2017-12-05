//
//  Manager.m
//  OpenCVTest
//
//  Created by RAJESH on 3/10/16.
//  Copyright © 2016 RAJESH. All rights reserved.
//

#import "IOSManager.h"
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>

using namespace std;

@implementation GrabCutManager

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

- (cv::Mat1b)cvMatMaskerFromUIImage:(UIImage *) image{
    
    // First get the image into your data buffer
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);
    
    //    cv::Mat1b markers((int)height, (int)width);
    //    markers.setTo(cv::GC_PR_BGD);
    cv::Mat1b markers = mask;
    uchar* data =  markers.data;
    
    int countFGD=0, countBGD=0, countRem = 0;
    
    for(int x = 0; x < width; x++){
        for( int y = 0; y < height; y++){
            NSUInteger byteIndex = ((image.size.width  * y) + x ) * 4;
            UInt8 red   = rawData[byteIndex];
            UInt8 green = rawData[byteIndex + 1];
            UInt8 blue  = rawData[byteIndex + 2];
            UInt8 alpha = rawData[byteIndex + 3];
            
            if(red == 255 && green == 255 && blue == 255 && alpha == 255){
                data[width*y + x] = cv::GC_FGD;
                countFGD++;
            }else if(red == 0 && green == 0 && blue == 0 && alpha == 255){
                data[width*y + x] = cv::GC_BGD;
                countBGD++;
            }else{
                countRem++;
            }
        }
    }
    
    free(rawData);
    
    NSLog(@"Count %d %d %d sum : %d width*height : %d", countFGD, countBGD, countRem, countFGD+countBGD + countRem, width*height);
    
    return markers;
}


-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

-(Mat3b) maskImageToMatrix:(CGSize)imageSize{
    int cols = imageSize.width;
    int rows = imageSize.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC3); // 8 bits per component, 4 channels (color channels + alpha)
    cvMat.setTo(0);
    
    uchar* data = mask.data;
    
    int fgd,bgd,pfgd,pbgd;
    fgd = 0;
    bgd = 0;
    pfgd = 0;
    pbgd = 0;
    
    for(int y = 0; y < rows; y++){
        for( int x = 0; x < cols; x++){
            int index = cols*y+x;
            if(data[index] == GC_FGD){
                cvMat.at<Vec3b>(cv::Point(x,y)) = Vec3b(255,0,0);
                fgd++;
            }else if(data[index] == GC_BGD){
                cvMat.at<Vec3b>(cv::Point(x,y)) = Vec3b(0,255,0);
                bgd++;
            }else if(data[index] == GC_PR_FGD){
                cvMat.at<Vec3b>(cv::Point(x,y)) = Vec3b(0,0,255);
                pfgd++;
            }else if(data[index] == GC_PR_BGD){
                cvMat.at<Vec3b>(cv::Point(x,y)) = Vec3b(255,255,0);
                pbgd++;
            }
        }
    }
    
    NSLog(@"fgd : %d bgd : %d pfgd : %d pbgd : %d total : %d width*height : %d", fgd,bgd,pfgd,pbgd, fgd+bgd+pfgd+pbgd, cols*rows);
    
    return cvMat;
}

-(Mat4b) resultMaskToMatrix:(CGSize)imageSize{
    int cols = imageSize.width;
    int rows = imageSize.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    cvMat.setTo(0);
    
    uchar* data = mask.data;
    
    int fgd,bgd,pfgd,pbgd;
    fgd = 0;
    bgd = 0;
    pfgd = 0;
    pbgd = 0;
    
    for(int y = 0; y < rows; y++){
        for( int x = 0; x < cols; x++){
            int index = cols*y+x;
            if(data[index] == GC_FGD){
                cvMat.at<Vec4b>(cv::Point(x,y)) = Vec4b(0,0,0,255);
                fgd++;
            }else if(data[index] == GC_BGD){
                cvMat.at<Vec4b>(cv::Point(x,y)) = Vec4b(255,255,255,255);
                bgd++;
            }else if(data[index] == GC_PR_FGD){
                cvMat.at<Vec4b>(cv::Point(x,y)) = Vec4b(0,0,0,255);
                pfgd++;
            }else if(data[index] == GC_PR_BGD){
                cvMat.at<Vec4b>(cv::Point(x,y)) = Vec4b(255,255,255,255);
                pbgd++;
            }
        }
    }
    
    NSLog(@"fgd : %d bgd : %d pfgd : %d pbgd : %d total : %d width*height : %d", fgd,bgd,pfgd,pbgd, fgd+bgd+pfgd+pbgd, cols*rows);
    
    return cvMat;
}


-(void) resetManager{
    mask.setTo(cv::GC_PR_BGD);
    bgModel.setTo(0);
    fgModel.setTo(0);
}

-(UIImage*) doGrabCut:(UIImage*)sourceImage foregroundBound:(CGRect)rect iterationCount:(int) iterCount{
    cv::Mat img=[self cvMatFromUIImage:sourceImage];
    cv::cvtColor(img , img , CV_RGBA2RGB);
    cv::Rect rectangle(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height);
    
    // GrabCut segmentation
    cv::grabCut(img,    // input image
                mask,      // segmentation result
                rectangle,   // rectangle containing foreground
                bgModel,fgModel, // models
                iterCount,           // number of iterations
                cv::GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground
    
    UIImage* resultImage = [self UIImageFromCVMat:[self resultMaskToMatrix:sourceImage.size]];
    

    
    return resultImage;
}

- (UIImage *)doActiveContour:(UIImage *)singleChannelImage {
    Mat mat;
    UIImageToMat(singleChannelImage, mat);
    
    Mat target(mat.rows, mat.cols, CV_8U);
    
    //1. Convert to gray scale.
    Mat grayscaledMat(mat.rows, mat.cols, CV_8U);
    cvtColor(mat, grayscaledMat, CV_RGBA2GRAY);
    
    //2. Do Guassian blur
    GaussianBlur(grayscaledMat, target, cv::Size(9, 9), 0);
    
    Mat binary(mat.rows, mat.cols, CV_8U);
    threshold(target, binary, 100, 255, THRESH_BINARY);
    
    
    Mat cannied(binary.rows, binary.cols, CV_8U);
    Canny(binary, cannied, 90, 150);
    
    //dilate(cannied, cannied, Mat(), cv::Point(1, -1));
    //dilate(cannied, cannied, Mat(), cv::Point(1, -1));
    
    //4. Create Contours & Get rectangular contour bounds.
    Mat contoursMat = cannied.clone();
    vector<vector<cv::Point>> contours;
    findContours(contoursMat, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    for (size_t i = 0; i < contours.size(); i++){
        cv::Rect r = boundingRect(contours[i]);
        if (r.width <= 30 && r.height <= 30){
            rectangle(cannied, r.tl(), r.br(), cv::Scalar(0, 0, 0), CV_FILLED);
        } else {
            //rectangle(cannied, r.tl(), r.br(), cv::Scalar(128, 128, 128));
        }
    }
    
    dilate(cannied, cannied, Mat(), cv::Point(1, -1));
    
    // Draw blue lines on original image.
    for (int y = 0; y < mat.rows; y++){
        for (int x = 0; x < mat.cols; x++){
            Vec4b pt = cannied.at<uchar>(y, x);
            if (pt[0] != 0) {
                Vec4b & target = mat.at<Vec4b>(y, x);
                target[0] = 255;   //0 - Blue ; 255 - red
                target[1] = 0; // 145 - Blue ; 0 - red
                target[2] = 0; // 146 - Blue; 0 - red
            }
        }
    }
    
    return MatToUIImage(mat);
}

// Active contour method Changed
#if 0
-(UIImage*) doActiveContour:(UIImage*)singleChannelImage
{
//  Mat src; Mat src_gray;
  int thresh = 100;
  int max_thresh = 255;
    RNG rng(12345);
  
  //cvMatGrayFromUIImage:cvImage;
  cv::Mat gray=[self cvMatFromUIImage:singleChannelImage];
  // Convert the image to grayscale
  cv::cvtColor(gray, gray, CV_RGBA2GRAY);
  // Apply Gaussian filter to remove small edges
  cv::GaussianBlur(gray, gray,
                   cv::Size(5, 5), 1.2, 1.2);
  // Calculate edges with Canny
  cv::Mat edges;
  cv::Canny(gray, edges, thresh, max_thresh, 3);
  // Fill image with white color
  cvImage.setTo(cv::Scalar::all(255));
  // Change color on edges
  cvImage.setTo(cv::Scalar(0, 128, 255, 255), edges);
  
  std::vector<cv::Vec4i> hierarchy;
  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(edges, contours, hierarchy ,cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  
  Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );
  for ( size_t i=0; i<contours.size(); ++i )
  {
    cv::drawContours( drawing, contours, int(i), Scalar(200,0,0), 1, 8, hierarchy, 0, cv::Point() );
    cv::Rect brect = cv::boundingRect(contours[i]);
    cv::rectangle(drawing, brect, Scalar(255,0,0));
  }

  // Convert cv::Mat to UIImage* and show the resulting image
  UIImage* resultImage = [self UIImageFromCVMat:drawing];
  
  return resultImage;
  
}
#endif

-(UIImage*) doGrabCutWithMask:(UIImage*)sourceImage maskImage:(UIImage*)maskImage iterationCount:(int) iterCount{
    cv::Mat img=[self cvMatFromUIImage:sourceImage];
    cv::cvtColor(img , img , CV_RGBA2RGB);
    
    cv::Mat1b markers=[self cvMatMaskerFromUIImage:maskImage];
    cv::Rect rectangle(0,0,0,0);
    // GrabCut segmentation
    cv::grabCut(img, markers, rectangle, bgModel, fgModel, iterCount, cv::GC_INIT_WITH_MASK);
    
    cv::Mat tempMask;
    cv::compare(mask,cv::GC_PR_FGD,tempMask,cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(img.size(),CV_8UC3,
                       cv::Scalar(255,255,255));
    
    tempMask=tempMask&1;
    img.copyTo(foreground, tempMask);
    
    UIImage* resultImage=[self UIImageFromCVMat:foreground];
    

    
    return resultImage;
}
@end
