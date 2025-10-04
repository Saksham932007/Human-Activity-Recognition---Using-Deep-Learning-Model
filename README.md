# Human Activity Recognition using Deep Learning 🚶‍♂️

This project implements a deep learning model to classify human activities based on sensor data. A stacked Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) is built using TensorFlow to recognize six different activities.

## 📜 Overview

Human Activity Recognition (HAR) is a well-known time-series classification problem. The goal is to predict a person's physical activity (like Walking, Sitting, Standing, etc.) using data collected from sensors, typically from a smartphone's accelerometer and gyroscope.

This project uses a supervised learning approach to train a robust classifier. The model architecture is designed to capture temporal dependencies in the sensor data, which is crucial for distinguishing between different activities.

-----

## 📊 Dataset

The model is trained on a dataset containing pre-processed sensor signals from a smartphone's accelerometer and gyroscope.

  * **Sensors Used:** Accelerometer and Gyroscope
  * **Data Shape:** The data is segmented into time windows, with each window having a fixed number of timesteps and features.
  * **Activities (Labels):** The model is trained to classify the following six activities:
      * Walking
      * Walking Upstairs
      * Walking Downstairs
      * Sitting
      * Standing
      * Laying

*(Note: You should add a link to the specific dataset you used here if it's publicly available.)*

-----

## 🧠 Model Architecture

The core of this project is a stacked LSTM network built with TensorFlow. LSTMs are well-suited for this task because of their ability to remember patterns over long sequences of data.

The model consists of the following layers:

1.  **Input Layer:** Expects data shaped as `(batch_size, time_steps, num_features)`.
2.  **Two Stacked LSTM Layers:** The model uses two hidden LSTM layers. This stacked architecture allows the model to learn higher-level temporal representations.
      * Hidden Units: `N_hidden_units`
      * Forget Bias: Set to `1.0` for stability.
3.  **Fully Connected Output Layer:** The output from the final LSTM layer is fed into a dense layer with weights and biases.
4.  **Softmax Activation:** A softmax function is applied to the output to produce a probability distribution across the activity classes.

The model also incorporates **L2 Regularization** to prevent overfitting by penalizing large weights. The loss is calculated using **Softmax Cross-Entropy**, and the model is optimized using the **Adam Optimizer**.

-----

## 🚀 Technologies Used

  * **Python 3.x**
  * **TensorFlow:** For building and training the deep learning model.
  * **Pandas:** For data manipulation and reading.
  * **NumPy:** For numerical operations.
  * **Scikit-learn:** For splitting the data into training and testing sets and for evaluation metrics.
  * **Matplotlib & Seaborn:** For data visualization, including plotting training progress and the confusion matrix.
  * **Jupyter Notebook / Google Colab:** For model development and experimentation.

-----

## 🔧 Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:** Create a `requirements.txt` file with the following content:

    ```
    pandas
    numpy
    tensorflow
    scikit-learn
    matplotlib
    seaborn
    ```

    Then, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**: Place the dataset files into a designated `data/` directory in the project's root folder.

-----

## 📈 How to Run

You can train the model and see the results by running the Jupyter Notebook:

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open the `.ipynb` file provided in the repository.
3.  Ensure the file paths for the dataset are correct.
4.  Run the cells in the notebook sequentially to train the model and visualize the results.

-----

## 🏆 Results and Evaluation

The model's performance is evaluated on the test set.

#### Training Progress

The following plots show the model's accuracy and loss on both the training and test datasets over the training epochs. This helps in diagnosing overfitting and understanding the model's convergence.



#### Confusion Matrix

A confusion matrix is used to visualize the model's performance on a per-class basis. The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier.

