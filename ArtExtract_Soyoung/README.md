<<<<<<< HEAD
# ArtExtract
=======
## ArtExtract 🎨 

#### Project overview
The ArtExtract project leverages machine learning to revolutionize art conservation by uncovering hidden paintings through multispectral imaging. By creating a comprehensive dataset of multispectral images of paintings, the project aims to develop an AI model capable of detecting hidden artworks behind the canvas. The project's innovative techniques could lead to significant discoveries in art history. Initially, the focus is on generating high-quality multispectral images from RGB images, addressing the challenges and strategic approaches needed to achieve this goal.

#### Dataset

| Dataset  | Info | Info |
| ------------- | ------------- |------------- |
| CAVE  | 32 scenes with full spectral resolution reflectance data from 400nm to 700nm at 10nm intervals (31 bands total)  | [Link](https://www.cs.columbia.edu/CAVE/databases/multispectral/)  |
| Harvard | 50 indoor and outdoor images, with 25 images featuring mixed illumination  | [Link](https://vision.seas.harvard.edu/hyperspec/d2x5g3/)  |

<!-- Will be updated -->
Link below will be updated soon...
[Download selected dataset for the training]()

[ Dataset Structure]
 ```               
├── train 
│   │
│   ├── rgb_images  # RGB images for training including objects and scenaries                      
│   └── ms_masks    # 8 multispectral images per 1 rgb image
└── val 
    ├── rgb_images  # Allocated painting image                       
    └── ms_masks  
```
#### Directories
```                
├── unets               # Collection of different UNets            
│   ├── transBlocks     # Chunks of Transformer blocks          
│   └── ..       
├── utils               # For the data loading, eval metrics and visualization
├── model.py            # Best performing model       
├── train.py            # Train, test code       
└── trainModel.ipynb    # Example 
```
#### Implementation guidance

1) Install the required packages
```
pip install -r requirements.txt
```
2) Train the model 
```
<!-- Example -->
python train.py --trainpath  '../train/' --valpath '../val/' -lr 0.02 -e 100
```
#### Results
- Final results will be released by the final evaluation period.

1. Quantitative Analysis
2. Qualitative Analysis

All the citations for the referred papers are cited at the top of each code base.

You can find out more about the project on the [blog]([https://medium.com/@soyoungpark.psy](https://medium.com/@soyoungpark.psy/beneath-the-canvas-discovering-hidden-art-with-ai-part1-gsoc-24-3dc499758120)) where I explained the step-by-step process of the project in detail.





>>>>>>> PRbranch
