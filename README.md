# VisualAI Q&A: Enabling Machines to Answer Images

[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

<!-- Project Logo -->
<p align="center">
  <img src="project_logo.png" alt="Project Logo" width="200" height="200">
</p>

## Overview

Welcome to VisualAI Q&A, a revolutionary project that empowers machines to answer questions about images. This project combines computer vision and natural language processing to tackle the challenging task of Visual Question Answering (VQA). VisualAI Q&A is your gateway to the exciting world of AI and deep learning.

#### Project Highlights

- **Seamless Fusion**: We seamlessly combine image understanding with language comprehension, enabling machines to answer questions about the content of images.
- **Cutting-Edge Models**: Utilizing state-of-the-art deep learning models such as VGG16 and LSTM for image and language understanding.
- **Interactive Visualization**: Engaging visualizations and interactive demos for users to explore the model's decisions.
- **Inclusivity**: We've created this project to be accessible and understandable for both beginners and experts in the fields of computer vision and natural language processing.

## Project Structure

Here's a glimpse of the project's structure:

|-- data # Data files and preprocessing scripts  

|-- checkpoints # Model checkpoints  

|-- notebooks # Jupyter notebooks for experimentation and development  

|-- src # Source code for the project  

|-- README.md # The document you're currently reading    



## How It Works

In a nutshell, VisualAI Q&A operates in a few simple steps:

1. **Data Preprocessing**: We've prepared scripts to download and preprocess the required data from the COCO dataset.
2. **Deep Learning Models**: We've implemented the key components of VQA: an image encoder, a question encoder, and a fusion model, powered by state-of-the-art deep learning architectures.
3. **Model Training**: Our models are trained using a dataset of images, questions, and answers, and fine-tuned to provide accurate answers.
4. **Validation and Visualization**: The accuracy of the model is validated, and we provide stunning visualizations to help you understand the model's decisions.


## Datasets

1. **Training Images**:
   - Download the training images from the following URL:
   - [Training Images](http://images.cocodataset.org/zips/train2014.zip)
   - Please note that this file is large, so ensure you have sufficient storage space and a stable internet connection for the download.

2. **Validation Images**:
   - Download the validation images from the following URL:
   - [Validation Images](http://images.cocodataset.org/zips/val2014.zip)
   - Similar to the training images, the validation images are also quite large.

3. **Training Questions**:
   - Access the training questions in a zip file:
   - [Training Questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip)

4. **Training Annotations**:
   - Download the training annotations in a zip file:
   - [Training Annotations](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip)

5. **Validation Questions**:
   - Find the validation questions in a zip file:
   - [Validation Questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip)

6. **Validation Annotations**:
   - Download the validation annotations in a zip file:
   - [Validation Annotations](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip)
## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/visualai-qa.git
   ```
2. Install Dependencies:

```bash
pip install -r requirements.txt
```
## Execution  

Execute the data download script we've provided to obtain the necessary dataset files.  


**Training the Model:**

Use the provided Jupyter notebooks to train the model from scratch or fine-tune it as needed.

## Validation and Visualization:

Further, explore the project by using our notebooks to validate the model's performance and dive into its attention mechanisms.

## Contributing
We invite contributors to help enhance VisualAI Q&A. Here's how you can become part of this exciting journey:

- Fork the Repository.
- Create a New Branch for your contributions.
- Submit a Pull Request.

## License
VisualAI Q&A is an open-source project and is made available under the MIT License.   
Feel free to use and adapt this project to meet your unique requirements.

## Contact
 please don't hesitate to reach out to the project's maintainers:  
 
Siva Keshav Yalamandala - sxy3510@mavs.uta.edu  

## Acknowledgments  

We'd like to express our deep appreciation to the project's contributors and the broader open-source community for making VisualAI Q&A possible. Together, we'll continue pushing the boundaries of AI!

