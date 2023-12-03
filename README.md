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
