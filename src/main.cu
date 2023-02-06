//
// Created by guidocalvano on 1/27/23.
//

#include "../include/main.cuh"
using namespace cv;

using namespace boost::filesystem;


void fileToBytes(std::string inputPathName, uint8_t* byteArray, int offset)
{
  //open file
  std::ifstream infile(inputPathName, std::ifstream::in);
  std::istreambuf_iterator<char> begin{infile}, end;

//get length of file
  uintmax_t size = file_size(inputPathName);
  printf("Size %u: ", size);
  for(int i = 0; i < offset; ++i) begin++;

  std::copy(begin, end, byteArray);
}

void bytesToGreyImageMat(Mat& img, int columns, int rows, uint8_t* imageData)
{
  img.create(columns,rows,CV_8UC1);

  for(int x = 0; x < columns; ++x)
  {
    for(int y = 0; y < rows; ++y) {
      img.at<uint8_t>(y, x) = imageData[y * columns + x];
    }
  }
}


void savePrediction(std::string path, unsigned int imageIndex, Mat img, int label)
{
  std::string filePathName = path +
      std::string("/image_") +
      std::to_string(imageIndex) +
      "_label_" +
      std::to_string(label) +
      ".png";
  imwrite(filePathName, img);
}

void savePredictions(std::string output_path,
                     int imageCount,
                     int rows,
                     int columns,
                     uint8_t* imageData,
                     uint8_t* labelData)
{
  Mat img;

  for(int i = 0; i < imageCount; ++i)
  {
    bytesToGreyImageMat(img, columns, rows, imageData + 28 * 28 * i);
    savePrediction(output_path, i, img, *(labelData + i));
  }
}


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

class Model
{
  cudnnHandle_t cudnn;                                    // cudnn interface
  cublasHandle_t cublas;

  // INPUT
  float *d_input{nullptr};
  float *d_dInput{nullptr};

  // CONVOLUTION LAYER
  // Convolution inputs
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnFilterDescriptor_t kernel_descriptor;

  // Convolution parameters
  cudnnConvolutionDescriptor_t convolution_descriptor;

  // Convolution itself
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  void *d_workspace{nullptr};

  float *d_kernel{nullptr};
  float *d_dKernel{nullptr};

  float *d_convolution_output{nullptr};
  float *d_dConvolution_output{nullptr};
  size_t workspace_bytes;
  // DENSE LAYER
  float* d_denseWeights;
  float *d_dDenseWeights{nullptr};

  // ACTIVATION FUNCTION
  float *d_activation;
  float *d_dActivation;
  cudnnActivationDescriptor_t activation_descriptor;

  float *d_output{nullptr};
  float *d_dOutput{nullptr};

  float* h_output;

  int columnCount;
  int rowCount;
  int colorCount;
  int labelClasses;

  public:
  Model(int columnCount, int rowCount, int colorCount, int labelClasses) {
    this-> columnCount = columnCount;
    this-> rowCount = rowCount;
    this->colorCount = colorCount;
    this->labelClasses = labelClasses;
    // input
    cudnnCreate(&cudnn);
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/1,
        /*channels=*/colorCount,
        /*image_height=*/rowCount,
        /*image_width=*/columnCount));

    // output
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/1,
        /*channels=*/colorCount,
        /*image_height=*/rowCount,
        /*image_width=*/columnCount));

    // kernel
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/colorCount,
        /*in_channels=*/colorCount,
        /*kernel_height=*/3,
        /*kernel_width=*/3));
    // DESCRIBE CONVOLUTION
    // convolution descriptor

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
        /*pad_height=*/1,
        /*pad_width=*/1,
        /*vertical_stride=*/1,
        /*horizontal_stride=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));

    checkCUDNN(
        cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
                                            &convolution_algorithm));

    // DEFINE MEMORY REQUIREMENTS
    workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes) << "B"
              << std::endl;

    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = colorCount * rowCount * columnCount * sizeof(float);

    cudaMalloc(&d_input, image_bytes);
    cudaMalloc(&d_dInput, image_bytes);

    cublasStatus status;

    status=cublasAlloc(rowCount*columnCount*colorCount, sizeof(float), (void**)&d_convolution_output);
    status=cublasAlloc(rowCount*columnCount*colorCount, sizeof(float), (void**)&d_dConvolution_output);

    status=cublasAlloc(rowCount*columnCount*colorCount * labelClasses,sizeof(float),(void**)&d_denseWeights);
    status=cublasAlloc(rowCount*columnCount*colorCount * labelClasses,sizeof(float),(void**)&d_dDenseWeights);

    status=cublasAlloc(labelClasses,sizeof(float),(void**)&d_activation);
    status=cublasAlloc(labelClasses,sizeof(float),(void**)&d_dActivation);

    cudaMalloc(&d_output, labelClasses * sizeof (float));
    cudaMalloc(&d_dOutput, labelClasses * sizeof (float));

    initWeightsEdgeDetector(colorCount, colorCount);

    // dense layer is matrix multiplaction
    // https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
    cublasCreate_v2(&cublas);

    // ACTIVATION FUNCTION
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
        /*mode=*/CUDNN_ACTIVATION_SIGMOID,
        /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
        /*relu_coef=*/0));
  }

  void initWeightsEdgeDetector(int kernelCount, int channelCount)
  {
    float h_kernel[kernelCount][channelCount][3][3];

    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMalloc(&d_dKernel, sizeof(h_kernel));

    // Mystery kernel
    const float kernel_template[3][3] = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}
    };

    for (int kernel = 0; kernel < kernelCount; ++kernel) {
      for (int channel = 0; channel < channelCount; ++channel) {
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            h_kernel[kernel][channel][row][column] = kernel_template[row][column];
          }
        }
      }
    }
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
  }

  void initWeightsRandomUniform(int kernelCount, int channelCount)
  {
    float h_kernel[kernelCount][channelCount][3][3];

    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMalloc(&d_dKernel, sizeof(h_kernel));

    for (int kernel = 0; kernel < kernelCount; ++kernel) {
      for (int channel = 0; channel < channelCount; ++channel) {
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            h_kernel[kernel][channel][row][column] = r * 2.0 - 1.0;
          }
        }
      }
    }
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
  }



  void deleteModel()
  {
    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cublasFree(d_convolution_output);
    cublasFree(d_dConvolution_output);

    cublasFree(d_denseWeights);
    cublasFree(d_dDenseWeights);

    cudnnDestroyActivationDescriptor(activation_descriptor);

    cublasDestroy_v2(cublas);
    cudnnDestroy(cudnn);
  }

  void backwardPass()
  {
    const float alpha = 1, beta = 0;

    // activation backprop
    cudnnStatus_t status = cudnnActivationBackward(
                                         cudnn,
                                         activation_descriptor,
                                         &alpha,
                                         output_descriptor,
                                         d_output,
                                         output_descriptor,
                                         d_dOutput,
                                         output_descriptor,
                                         d_activation  ,
                                         &beta,
                                         output_descriptor,
                                         d_dActivation);
    // backpropagate to previous layer
    // C = alpha * op(A) * op(B) + beta * C
    cublasSgemm('n', // op(A) = A
                't', // op(B) = transpose(B)
                1, //rows in C and A
                columnCount * rowCount * colorCount, // Columns in C and B
                labelClasses, // columns in A and rows in B, i.e. the dimensions that are dot producted away
                1,
                d_dActivation,
                1,
                d_denseWeights,
                columnCount * rowCount * colorCount,
                0,
                d_dConvolution_output,
                1);

    // backpropagate to weights

    cublasSgemm('n',
                'n',
                columnCount * rowCount * colorCount,
                labelClasses,
                1,
                1,
                d_convolution_output,
                columnCount * rowCount * colorCount,
                d_dActivation,
                1,
                0,
                d_dDenseWeights,
                columnCount * rowCount * colorCount);
    status = cudnnConvolutionBackwardFilter(
                                            cudnn,
                                             &alpha,
                                            input_descriptor,
                                         d_input,
                                         output_descriptor,
                                         d_dConvolution_output,
                                convolution_descriptor,
                                            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                            d_workspace,
                                            workspace_bytes,
                                      &beta,
               kernel_descriptor,
                                            d_dKernel);

    status =  cudnnConvolutionBackwardData(
                                               cudnn,
                                               &alpha,
                                               kernel_descriptor,
                                               d_kernel,
                                               output_descriptor,
                                               d_dConvolution_output,
                                               convolution_descriptor,
                                               CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                               d_workspace,
                                               workspace_bytes,
                                               &beta,
                                               input_descriptor,
                                               d_dInput);



  }

  void forwardPass(float* imageData)
  {

    // forward pass through convolution
    cudaMemcpy(d_input, imageData, rowCount * columnCount * colorCount * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemset(d_output, 0, image_bytes);

    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_convolution_output));
    // dense layer is matrix multiplication.
    // https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.htmls
    cublasSgemm('n','n',1,labelClasses, columnCount * rowCount * colorCount,1,d_convolution_output,1,d_denseWeights,columnCount * rowCount * colorCount,0,d_activation,1);

    // Perform the forward pass of the activation
    checkCUDNN(cudnnActivationForward(cudnn,
                                      activation_descriptor,
                                      &alpha,
                                      output_descriptor,
                                      d_output,
                                      &beta,
                                      output_descriptor,
                                      d_output));


    // COPY MEMORY TO HOST
    h_output = new float[columnCount * rowCount * colorCount * sizeof(float)];
    cudaMemcpy(h_output, d_output, columnCount * rowCount * colorCount * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory and copy info back to host

  }

  void computeLoss(int correctPrediction)
  {
    float h_dOutput[labelClasses];

    for(int l = 0; l < labelClasses; l++)
    {
      h_dOutput[l] = -h_output[l];

      if(l == correctPrediction)
      {
        h_dOutput[l] = 1.0 - h_output[l];
      }
    }

    cudaMemcpy(d_dOutput, h_dOutput, labelClasses * sizeof(float), cudaMemcpyHostToDevice);
  }

  int outputToPrediction()
  {
    float highestProbability = -1.0;
    int highestLabel = -1;

    for(int l = 0; l < labelClasses; l++)
    {
      float probability = h_output[l];
      if(probability > highestProbability)
      {
        highestProbability = probability;
        highestLabel = l;
      }
    }
    return highestLabel;
  }

};

void run(int imageCount, int columnCount, int rowCount, int colorCount, int labelClasses, uint8_t* imageData, uint8_t* labelData, uint8_t* predictionData)
{
  float* h_imageData = (float*) malloc(imageCount * rowCount * columnCount * colorCount * sizeof(float));

  for(int i = 0; i < imageCount; ++i)
    for(int x =0; x < columnCount; ++x)
      for(int y=0; y < rowCount; ++y)
        for(int c=0; c < colorCount; ++c)
          h_imageData[i * columnCount * rowCount * colorCount +
                  y * columnCount * colorCount + x * colorCount + c] =
                      (float) imageData[i * columnCount * rowCount * colorCount +
                      y * columnCount * colorCount + x * colorCount + c];

  float* d_imageData{nullptr};
  cudaMalloc(&d_imageData, sizeof(h_imageData));
  cudaMemcpy((void*) d_imageData, (void*) h_imageData, imageCount*rowCount*colorCount*columnCount * sizeof(float), cudaMemcpyHostToDevice);

  Model model(columnCount, rowCount, colorCount, labelClasses);

  model.forwardPass(d_imageData);

  int prediction = model.outputToPrediction();

  model.computeLoss(labelData[0]);

  model.backwardPass();
  return;

}

void loadTrainingData(uint8_t* imageData, uint8_t* labelData)
{
//   std::vector<uint8_t> imageByteVector;
//   std::vector<uint8_t> labelByteVector;
  int imageOffset = 4 * 4;
  int labelOffset = 2 * 4;
   fileToBytes("./data/input/training_image.idx3", imageData, imageOffset);
   fileToBytes("./data/input/training_label.idx1", labelData, labelOffset);

//
//   std::cout << " image byte vector size "<< imageByteVector.size() - imageOffset << std::endl;
//   std::cout << " label byte vector size "<< labelByteVector.size() - labelOffset << std::endl;
//
//   std::memcmp(imageData, imageByteVector.data() + imageOffset, imageByteVector.size() - imageOffset);
//   std::memcmp(labelData, labelByteVector.data() + labelOffset, labelByteVector.size() - labelOffset);
}


int main(int argc, char *argv[])
{
  if(argc < 3) {
      printf("Syntax: main.exe inputpath outputpath");
      return 1;
  }

  // cudaError_t err = cudaDeviceReset();
  // hardcoded image data because c++ is acting difficult
  int32_t imageCount = 60000;
  int32_t columns = 28;
  int32_t rows = 28;
  int32_t labelClasses = 10;
  int colorCount = 1;

  // allocating this as arrays on the stack causes a segfault, because apparently the stack doesn't like it if you
  // allocate 47MB on it...
  uint8_t* imageData = new uint8_t[rows*columns*imageCount*colorCount];
  uint8_t* labelData = new uint8_t[imageCount];
  uint8_t* predictionData = new uint8_t[imageCount];

  // this will hideously crash if you don't have the right dimensions but my goal is to just explore cudnn so I don't care
  loadTrainingData(imageData, labelData);

  run(imageCount, columns, rows, colorCount, labelClasses, imageData, labelData, predictionData);

  savePredictions(std::string("./data/output"), imageCount, rows, columns, imageData, labelData);//, labelClasses);
  // err = cudaDeviceReset();
  delete imageData;
  delete labelData;
  delete predictionData;
}