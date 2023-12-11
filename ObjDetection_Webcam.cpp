#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <fstream> 

/*
The fstream class is part of the C++ Standard Library and provides functionality for reading from and writing to files.

Here are some key components related to fstream:

ifstream (Input File Stream): This class is used for reading from files. You can create an object of ifstream to read data from a file.
ofstream (Output File Stream): This class is used for writing to files. You can create an object of ofstream to write data to a file.
fstream (File Stream): This class can be used for both reading and writing to files. You can create an object of fstream to perform both input and output operations on a file.
*/

using namespace cv; // For CV 
using namespace std; // To avoid std::string or cv2::imread
using namespace cv::dnn; // For neural net stuff

//// FOR EVERY PROJECT: Have a CMakeLists.txt with the name of the project and do cmake .. or the path to the cmakelist.txt 
/// Then once we have the code, do make and then ./NAMEPROJECT to run it. 


/// PROJECT: USING YOLOV5 WITH C++

/// Definition of variables - (1,C,H,W) of NN with C = 3 and 1 is the batch size. We use a BLOB -> binary large object, containing the data in raw format.

const float INPUT_WIDTH = 640.0; // Const type is to make it fixed, can't change its value.
const float INPUT_HEIGHT = 640.0;

const float SCORE_THRESHOLD = 0.5; // To filter low probability class scores.
const float NMS_THRESHOLD = 0.45; //  To remove overlapping bounding boxes.
const float CONFIDENCE_THRESHOLD = 0.45; // Filters low probability detections.

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Some colours to represent bounding boxes I guess. 

Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);


void display_img(Mat img){

    imshow("Image test", img);
    waitKey(0); // Infinity wait is when you set it to 0.

}

void draw_label(Mat img, string label, int left, int top){ // This takes the image, the label text, and the TL points (left,x) and (top, y) coordinates.

    int baseLine; // This calculates the height of the text of the label
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine); // We use this function to know the size in x and y of the text given the string and its properties.
                                                                                       // &baseLine is a pointer. It is not defined yet with a value, but getTextSize gives us the value. Probably we can't do label_size, baseline = .... and that why we need to do it like that?
                                                                                       // label_size will have height and width as objects. Not so sure what is baseline so play with it once we have results.
    top = max(top, label_size.height); // This ensures that the text does not overlap with the bounding box. 
    Point Point_TL = Point(left,top); // (x,y)
    Point Point_BR = Point(left+label_size.width,top+label_size.height+baseLine); // PLAY WITH BASELINE. 

    rectangle(img, Point_TL,Point_BR, BLACK,FILLED);
    putText(img, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, BLUE, THICKNESS);


}

// The function pre–process takes the image and the network as arguments. At first, the image is converted to a blob. Then it is set as input to the network. 
// The function getUnconnectedOutLayerNames() provides the names of the output layers. 
// It has features of all the layers, through which the image is forward propagated to acquire the detections. After processing, it returns the detection results.

vector<Mat> Preprocessing(Mat img, Net network){

    Mat blob;
    vector<Mat> predictions; // An vector of arrays telling the outputs (x,y,w,h,confidence,labelclass)

    blobFromImage(img, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false); // Input, blob output, normalisation (everything 0-1), resize image to the input layer size, Scalar() means no mean subtraction -> it would subtract the mean of each channel to its corresponding channel. First TRUE means swap BGR to RGB and second FALSE means image is NOT cropped after resizing
 
    network.setInput(blob); // This method sets the input blob for the neural network model. It means that the preprocessed image (or data) contained in the blob will be fed as input to the neural network for inference or forward-pass.
 
    // Forward propagate.
    network.forward(predictions, network.getUnconnectedOutLayersNames()); // This provides the names of the unconnected output layers to the forward method, indicating which layers' outputs should be considered as the final predictions.

    return predictions;

}




Mat Postprocessing(Mat img, vector<Mat> results, const vector<string> class_names){

    vector<int> Class_ID;
    vector<float> Confidence_val;
    vector<Rect> Boundingbox;
    vector<int> indices;

    // Outputs is of size 25200x85 by default, so 25200 detections with x,y,w,h,confidence, classIDvalue for each detection. 
    // X,Y represent the centre coordinates of the bounding box, and W and H its width and height.

    float x_factor = img.cols / INPUT_WIDTH; // This is done because maybe the input does not have the 640/640 size we require, so then this is modified accordingly to put the label where it corresponds.
    float y_factor = img.rows / INPUT_HEIGHT; 

    float *data = (float *)results[0].data; // Feels like we are flattening the 25200x85 matrix.
    const int dimensions = 85; // because we have 80 classes + 5 extra parameters (x,y,w,h,classes)
    
    const int rows = 25200; // 25200 for default size 640

    for(int i=0; i< rows; i++){ // We iterate through all the detections. 

        float Confidence = data[4]; // The fifth element conotains the confidence level of the detection. 

        if(Confidence >= CONFIDENCE_THRESHOLD){ // We need to be confident about the detection location to proceed.

            float * class_scores = data + 5; // In the provided code, data + 5 is used to create a pointer to the element at index 5 in the array pointed to by data. 
                                             // This operation effectively skips the first 5 elements of the array and points to the 6th element. 
                                             // This is often used when working with data structures where the initial elements contain information that is not needed at a particular point in the code.

            Point max_class_ID; // Not necessarily need to be a point, but stores index of class with maximum confidence. 
            double max_class_score; // We store the maximum value of confidence with double precision
            Mat scores(1, class_names.size(), CV_32FC1, class_scores); // Create a 1x80 Mat and store class scores of 80 classes.
          
            minMaxLoc(scores, 0, &max_class_score, 0, &max_class_ID); // OpenCV function to calculate min,max value and its index location. We pass the matrix, 2nd/4th elements are min value and min location (set to 0 to not return it) and 3/5 elemenets same for max.

            if (max_class_score > SCORE_THRESHOLD){ // We need to be confident with the class score to proceed

                Confidence_val.push_back(Confidence); // Retrieve confidence value
                Class_ID.push_back(max_class_ID.x); // I guess y will just be 1

                float Cx = data[0];
                float Cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((Cx-w*0.5)*x_factor);
                int top = int((Cy-h*0.5)*y_factor);
                int width = int(w*x_factor);
                int height = int(h*y_factor);

                Boundingbox.push_back(Rect(left,top,width,height));
            }
        }
        data += 85; // Since each detection is 85 elements, we sum 85 so that the first element is x from the next detection and start loop again. 
    }

    cout << Boundingbox.size() << endl; 
    
    NMSBoxes(Boundingbox, Confidence_val, SCORE_THRESHOLD, NMS_THRESHOLD, indices); // The purpose of non-maximum suppression is to eliminate redundant and overlapping bounding boxes, keeping only the most confident ones. 
                                                                                    // The algorithm compares the confidence scores of bounding boxes and their overlaps (IoU), discarding those that fall below the specified thresholds.
    cout << indices.size() << endl;                                                                                // Indices is a vector of the bounding boxes IDs that survive the non-maximum suppression algorithm. NMS_THRESHOLD: anything above this value in IoU is considered overlapped Bounding box.
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i]; // We get the indices of the surviving bounding boxes and get the box/labels from those indices. 
        Rect box = Boundingbox[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(img, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", Confidence_val[idx]); // Confidence value with 2 decimal precision. 
        label = class_names[Class_ID[idx]] + ":" + label; // The full label is the class belonging to the index ID.
        // Draw class labels.
        draw_label(img, label, left, top);
    }
    
    return img;

}



int main()
{
   vector<Mat> results;
   Mat img, img_resized, img_processed;
   Net network;
   vector<string> class_list;
   string line;
   string path = "/Users/german/Library/CloudStorage/Dropbox/Projects/CV_Cpp_ObjDetection_YOLOv5/YOLOv5/data/images/zidane.jpg";
   vector<double> layersTimes;
   
   VideoCapture cap(0); // For a webcam, instead of a path, you use the webcamID (0 if only have one webcam)

   ifstream ifs("/Users/german/Library/CloudStorage/Dropbox/Projects/CV_Cpp_ObjDetection_YOLOv5/YOLOv5/data/coco.names"); // This class is used for reading from files. You can create an object of ifstream to read data from a file and we associatte the variable to ifs.

   while (getline(ifs, line)) // The getline function is used to read a line from the input file stream (ifs). It takes two parameters: ifs: The input file stream from which the line is read.
                              // line: A string variable where the read line is stored.
   {    
      class_list.push_back(line); // We append each class in the string vector. 
   }
   
   network = readNet("/Users/german/Library/CloudStorage/Dropbox/Projects/CV_Cpp_ObjDetection_YOLOv5/YOLOv5/models/YOLOv5s.onnx");


   while(true){
      cap.read(img);
      resize(img, img_resized,Size(), 1,1); // To escale it a specific factor.


      results = Preprocessing(img_resized, network);
      img_processed = Postprocessing(img_resized, results, class_list);
  
      double freq = getTickFrequency() / 1000;
      double t = network.getPerfProfile(layersTimes) / freq;
      string label = format("Inference time : %.2f ms", t);
      putText(img_processed, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED,2);
      display_img(img_processed);
      if (waitKey(1) == 27) // If we press escape (27) we close the window of the webcam 
         break; 
   }
   destroyAllWindows();


}

