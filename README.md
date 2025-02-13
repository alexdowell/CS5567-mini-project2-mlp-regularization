# CS5567 Mini Project 2: MLP Regularization for Regression  

## Description  
This repository contains **Mini Project 2** for **CS5567**, focusing on the application of **multilayer perceptron (MLP) neural networks** for regression tasks and the effects of **regularization techniques** to mitigate overfitting. The project is divided into two main tasks:  

- **Task A:** Correlation-based feature selection and linear regression.  
- **Task B:** MLP neural network implementation, comparison of architectures, and the impact of regularization techniques.  

## Files Included  

### **Task A: Linear Regression and Feature Selection**  
- **File:** CS5567_mini_project2_A_B_final.m  
- **Topics Covered:**  
  - Computing correlation coefficients to determine relevant features.  
  - Implementing **linear regression models** for regression analysis.  
  - Performance evaluation using **RMSE and R² metrics**.  

### **Task B: MLP Model Implementation and Regularization Effects**  
- **File:** CS5567_mini_project2_A_B_final.m  
- **Topics Covered:**  
  - **Training an MLP with different hidden layer sizes** (2, 10, and 50 nodes).  
  - Splitting data into training and validation sets (80-20 and 30-70).  
  - Evaluating overfitting risks in deep networks.  
  - Applying **L2 weight decay regularization** to reduce overfitting.  

### **Project Report and Results**  
- **File:** CS5567_MiniProject2_slides.pptx  
- **Contents:**  
  - Summary of **linear regression findings**.  
  - Analysis of **MLP models with different architectures**.  
  - Interpretation of **regularization impacts on training and validation performance**.  

## Installation  
Ensure **MATLAB** is installed before running the scripts.  

### Required MATLAB Toolboxes  
- Deep Learning Toolbox  
- Statistics and Machine Learning Toolbox  

## Usage  
1. Open **MATLAB**.  
2. Load the required dataset (`bodyfat_dataset`).  
3. Run the script:  
   - `CS5567_mini_project2_A_B_final.m`  
4. Review **training performance**, **validation loss**, and **regularization effects** in MATLAB figures and console outputs.  

## Example Output  

- **Linear Regression Results (Task A):**  
  - RMSE: **4.47**  
  - R²: **0.719**  
  - Test MSE: **21.403**  

- **MLP Model Comparisons (Task B):**  
  - **10-node MLP:**  
    - Training MSE: **16.06**  
    - Validation MSE: **27.38**  
  - **50-node MLP (No Regularization):**  
    - Training MSE: **5.23**  
    - Validation MSE: **98.79** (Overfitting detected)  
  - **50-node MLP (Regularization = 0.5):**  
    - Training MSE: **9.87**  
    - Validation MSE: **129.74**  

## Contributions  
This repository is intended for **educational purposes** in **machine learning and regression analysis**. Feel free to fork and modify the project.  

## License  
This project is open for **academic and research use**.  

---
**Author:** Alexander Dowell  
