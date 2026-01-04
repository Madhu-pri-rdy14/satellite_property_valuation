Overview

This project implements a multimodal regression pipeline that combines tabular real-estate data with satellite imagery to predict house prices.
By integrating structured property attributes with visual neighbourhood context, the model captures both physical and environmental factors influencing property valuation.

Performance Summary

The following results were obtained after model training and evaluation:

Tabular-Only Model

R² Score: 0.8616

RMSE: $144,618

While the tabular model achieves higher numerical accuracy, the multimodal model provides better spatial awareness and interpretability, as validated using Grad-CAM visualizations.

Model Architecture

The architecture consists of two parallel branches that are fused for final price prediction:

CNN Branch (ResNet-18):
Processes 224×224 satellite images and extracts a 512-dimensional visual feature embedding.

MLP Branch:
Processes tabular property features (such as sqft, bedrooms, bathrooms, etc.) into a 64-dimensional feature vector.

Fusion Layer:
Concatenates image and tabular features into a unified representation, followed by fully connected layers for final house price regression.

Data Collection

Satellite images are fetched dynamically using latitude and longitude coordinates via the ArcGIS World Imagery API and are not stored in the repository to reduce repository size.

Explainability

Grad-CAM is used to interpret model predictions by highlighting regions in satellite images that most influence price estimation.
The visualizations show that the model focuses on road connectivity, building density, and neighbourhood structure, confirming that meaningful spatial features are learned.

Setup & Installation
Environment Requirements

Python 3.12.5 virtual environment

Development environment: VS Code

Required libraries are listed and imported in the Jupyter notebooks

GPU-enabled PyTorch was used for faster training (ensure the correct CUDA version is installed)

How to Run

Preprocessing

Ensure house prices are log-transformed using
log(1 + price) to stabilize training.

Training

Run the training notebook for 20 epochs

Learning rate: 1e-4

Epochs can be increased if higher R² is desired

Grad-CAM Visualization

Use the generate_final_gradcam function to visualize what regions of the satellite image influence predictions.

Outputs

Final predictions are provided in 24114097_final.csv

The CSV strictly follows the format:

id, predicted_price

Conclusion

This project demonstrates that combining satellite imagery with structured real-estate data enables more interpretable and location-aware property price prediction, highlighting the effectiveness of multimodal deep learning for real-world valuation tasks.
