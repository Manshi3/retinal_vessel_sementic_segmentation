# retinal_vessel_sementic_segmentation

Deep learning, in particular, semantic segmentation, has found its niche in medical imaging. From brain tumors to skin lesions, the applications of semantic segmentation in medical imaging are countless. For our journey into semantic segmentation + medical imaging, we will start with a small dataset. That’s not to say that the dataset is simple to solve. It has its intricacies, caveats, and complications. All of which we will cover while discussing the dataset in detail.

<h3> These are the points that we will cover in the article: </h3>
-> First, we will discuss the DRIVE (Digital Retinal Images for Vessel Extraction) dataset that we will use in the article.
-> After that, we will move on to the training experiments. We will carry out the following 4 experiments:
      -Training DeepLabV3 ResNet50 model with 512×512 images.
      -Training DeepLabV3 ResNet50 with 768×768 images.
      -DeepLabV3 ResNet101 training with 512×512 images.
      -DeepLabV3 ResNet101 training with 768×768 images.
-> After analyzing all the results, we will use the best model for running inference on a held-out test set.

<h1> The DRIVE Dataset for Retinal Vessel Segmentation </h6>

You can find the DRIVE dataset with training/validation split [here on Kaggle](https://www.kaggle.com/datasets/sovitrath/drive-trainvalidation-split-dataset). We will use this version of the dataset for training and evaluating the model.

After downloading and extracting the dataset, you will find the following structure:
![image](https://github.com/Manshi3/retinal_vessel_sementic_segmentation/assets/105007863/9bb7e57f-1fe8-42aa-9aa7-11429e171a31)

<h1>Directory Structure for Retinal Vessel Segmentation Project</h1>

![image](https://github.com/Manshi3/retinal_vessel_sementic_segmentation/assets/105007863/7d7a7f5f-3b3b-4377-b576-b24d3406164c)

<h4>All training and inference experiments were carried out on a machine with 10 GB RTX 3080 GPU, 10th generation i7 CPU, and 32 GB of RAM.
else use GOOGLE COLABORATORY PLUS for the working of the model</h4>

<h2>Training DeepLabV3 ResNet50 with 512×512 Resolution</h2>
We will start with training the DeepLabV3 ResNet50 with 512×512 resolution. This is the base resolution for this dataset and we will not go below this. It’s very difficult to get good results for this dataset with lower resolution images.

                  python train.py --epochs 100 --lr 0.0001 --batch 2 --imgsz 512

<h2>Training DeepLabV3 ResNet50 with 768×768 Resolution</h2>
Let’s move on to the next experiment. This time we will train the same DeepLabV3 ResNet50 model but with 768×768 resolution.

Note: This training experiment requires quite a lot of VRAM, ~ 8 GB.

To start the training experiment, we can execute the following command.

                  python train.py --epochs 100 --lr 0.0001 --batch 2 --imgsz 768

<h2>Training DeepLabV3 ResNet101 with 512×512 Resolution</h2>
Now, we will use the DeepLabV3 ResNet101 for training on the Retinal Vessel Segmentation dataset.

If you are also training your own models while following the article, please uncomment the function in model.py which loads the deeplabv3_resnet101 model and comment out the previous function.

The training command is going to be the same as the first experiment as the only change we needed was in the model.py file.

                  python train.py --epochs 100 --lr 0.0001 --batch 2 --imgsz 512

<h2>Training DeepLabV3 ResNet101 with 640×640 Resolution</h2>
We are down to the final training experiment. We will train the DeepLabV3 ResNet101 model with 640×640 resolution.

But why 640×640 and not 768×768? It’s only because of memory constraints. I could only fit 640×640 resolution images into 10 GB RTX 3080. In case you are a GPU-savvy person, it consumed exactly 9.8 GB out of the 10 GB with 2 Chrome tabs open. On a side (and fun) note, I could not carry out the training when 10 Chrome tabs were open. There was not enough VRAM available in that case.

This is the command to start the training.

                  python train.py --epochs 100 --lr 0.0001 --batch 2 --imgsz 640

<h2>Inference using the Best Model</h2>

We will run the inference on the test images that come with the dataset. We will use the inference_image.py script for this.

                  python inference_image.py --model ../outputs/best_model_iou.pth --input ../input/DRIVE_train_val_split/test_images/ --imgsz 768




