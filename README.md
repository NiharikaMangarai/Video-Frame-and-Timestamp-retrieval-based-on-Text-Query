# Video-Frame-and-Timestamp-Retrieval-Based-on-Text-Query

This project aims to create a reliable system for content-based retrieval that can precisely retrieve video frames using textual queries. Our goal is to use deep learning techniques to create a semantic space and map both textual queries and video frames in the same space where the similarity between text and images can be measured directly.

We use ResNet50 and DistilBERT as our image encoder and text encoder respectively. Additionally, we used a loss function based on contrastive learning to maximize the alignment of modalities inside a common embedding space. Model performance testing on Flickr datasets and the benchmark MSRVTT dataset was conducted to calculate recall and other performance metrics. Lastly, a Streamlit application that accepts YouTube video links and text queries allows for testing demos in a real-world scenario. We have incorporated contrastive cross-entropy loss calculation thus obtaining positive-negative pair similarities. Contrastive cross-entropy loss enhances model performance by encouraging similar embeddings for similar inputs and dissimilar embeddings for distinct inputs, facilitating better representation learning in tasks like similarity comparison or clustering.

## Table of Contents

1. [Experiments & Implementation Details](#experiments--implementation-details)
   - [Pre-trained ResNet50](#pre-trained-resnet50)
   - [Pre-trained DistilBERT](#pre-trained-distilbert)
   - [Model Complexity](#model-complexity)
   - [Training Hyperparameters](#training-hyperparameters)
   - [Training Loop](#training-loop)
2. [Results and Performance](#results-and-performance)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Flickr Testing and Results](#flickr-testing-and-results)
   - [MSRVTT Testing Results](#msrvtt-testing-results)
3. [Streamlit Demo Application](#streamlit-demo-application)
   - [Application Details](#application-details)
   - [Performance Metrics](#performance-metrics)

## Experiments & Implementation Details

### Pre-trained ResNet50

We used a pre-trained ResNet50 with its robust feature extraction capabilities learned from ImageNet. By removing its last layers, we retained the convolutional base while discarding the classification head, enabling us to obtain image features. This process allows us to leverage the pre-trained ResNet50's ability to capture high-level features and spatial information from images.

### Pre-trained DistilBERT

DistilBERT has 40% fewer parameters than BERT-base-uncased, runs 60% faster, while preserving over 95% of BERT’s performance. It was pre-trained on the same corpus as the original BERT model, a concatenation of English Wikipedia and Toronto Book Corpus. DistilBERT retains 97% of BERT's performance on the GLUE benchmark.

### Model Complexity

Using multiple modalities does not significantly impact the number of parameters. Most of the parameters correspond to the BERT caption encoding module. The number of parameters of a transformer encoder is independent of the number of input embeddings, as are the parameters of a CNN from the image size, like ResNet50.

- **Total parameters**: 90.32M
  - **Caption encoder**: 67M (Projections: 0.675M, DistilBERT: 66.36M)
  - **Image encoder**: 24.3M (Projections: 0.788M, ResNet50: 25.5M)

Pre-trained models have already learned meaningful representations and may not require as much regularization during fine-tuning.

### Training Hyperparameters

- **Shared Embedding Size**: 512
- **Dropout Rate**: 0.1-0.2
- **Temperature Value**: 1 and 0.2
- **Maximum Sequence Length**: Defined during tokenization
- **Number of Epochs**: 25
- **Batch Size**: Defined based on the fine-tuning task

### Training Loop

1. **Forward pass**: Compute the projections for both image and text inputs.
2. **Compute Loss**: Calculate the contrastive clip loss function based on the projections.
3. **Backpropagation**: Update the model parameters by backpropagating the loss gradients.
4. **Loss Monitoring**: Track and monitor the training loss to assess model performance.
5. **Learning Rate Scheduling**: Adjust the learning rate based on the observed loss trends to facilitate convergence.

## Results and Performance

### Evaluation Metrics

R@1, R@5, and R@10 are evaluation metrics commonly used in information retrieval tasks, including video frame retrieval. They measure the accuracy of a retrieval system by assessing whether the correct item (in this case, video frame) is retrieved within the top 1, 5, or 10 ranks, respectively.

### Flickr Testing and Results

After training, we tested our model on the Flickr datasets to evaluate its performance:

- **Flickr8k**: Recall @1: 82.83, Recall @5: 94.68, Recall @10: 97.33
- **Flickr30k**: Recall @1: 83.66, Recall @5: 95.02, Recall @10: 98.46

### MSRVTT Testing Results

Testing on the MSRVTT dataset showed the following performance metrics:

- **Recall @1**: 52.86
- **Recall @5**: 72.56
- **Recall @10**: 79.50

## Streamlit Demo Application

### Application Details

Our Streamlit application allows users to input a YouTube video link and a text query to extract relevant frames from the video based on that query.

1. **User Input**: Users can input a YouTube video link and a query related to the content of the video.
2. **Frame Extraction**: The app downloads the specified YouTube video and extracts frames from it. It uses a pre-trained ResNet model to generate image embeddings for each frame.
3. **Text Embeddings**: The app generates text embeddings for the user's query using a pre-trained DistilBERT model.
4. **Frame Retrieval**: It calculates the similarity between the text embeddings of the query and the image embeddings of each frame. Based on this similarity score, it selects the top frames that best match the query.
5. **Display**: The selected frames are displayed along with their timestamps. Additionally, the app provides the option to view the entire video with marked timestamps corresponding to the selected frames.

### Performance Metrics

In our project, we implemented various performance metrics to assess the efficiency and effectiveness of our video frame retrieval system in real-world scenarios:

- **Downloading Speed**: Speed at which the YouTube video is downloaded.
- **Extraction Speed**: How fast frames are extracted from the downloaded video.
- **Frames per Second (FPS)**: Number of frames processed per second during extraction.
- **CPU Usage**: Percentage of CPU resources utilized during extraction.
- **Memory Usage**: Amount of memory consumed during operation.
- **Frames Processed**: Total number of frames processed by the system.

These metrics provide valuable insights into the efficiency, scalability, and resource requirements of our video frame retrieval system.

### Example 
#### 1

![image](https://github.com/NiharikaMangarai/Video-Frame-and-Timestamp-retrieval-based-on-Text-Query/assets/75827294/01650ef0-2670-46d5-880c-935da6cc9804)

#### 2
![image](https://github.com/NiharikaMangarai/Video-Frame-and-Timestamp-retrieval-based-on-Text-Query/assets/75827294/c908438f-aec5-449c-96ae-5ef52e9287b1)


