# Spam Message Detector

A machine learning-powered web application to detect and filter out spam messages.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Author](#author)
- [License](#license)

## Overview

This project is a simple yet powerful tool designed to classify WhatsApp messages as either "spam" or "ham" (not spam). The application is built with a machine learning model and a web interface, allowing users to enter a message and get an instant prediction.

## How It Works

The core of this application is a machine learning pipeline saved as a `scikit-learn` model. The pipeline includes:

1.  **Vectorizer**: A `TfidfVectorizer` that converts text messages into numerical features.
2.  **Classifier**: A `Multinomial Naive Bayes` classifier trained on a labeled dataset of spam and ham messages.

The model is deployed via a simple Flask web service, which handles user requests, makes a prediction using the loaded model, and returns the result to the user.

## Project Structure

The repository is organized as follows:

.
├── app.py                      # The Flask web application
├── spam_detection_pipeline.pkl # The saved machine learning model
├── requirements.txt            # Python dependencies for the project
├── start.sh                    # Shell script to run the app on Render
├── static/
│   ├── css/
│   │   └── style.css           # Custom styles for the web interface
│   └── images/
│       └── my_photo.jpg        # Author photo and other assets
└── templates/
├── index.html              # The main page with the message form
└── result.html             # The page that displays the prediction


## Technologies Used

* **Python**: The main programming language for the backend.
* **Flask**: A micro web framework for the web interface.
* **scikit-learn**: For machine learning, including vectorization and classification.
* **Joblib**: For saving and loading the machine learning pipeline.
* **HTML & CSS**: For designing the user interface.
* **Gunicorn**: A production-grade WSGI server for deployment.
* **Render**: The cloud platform used for continuous deployment.

## Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/TheCreativeLad/Spam-Detection.git](https://github.com/TheCreativeLad/Spam-Detection.git)
    cd Spam-Detection
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask application**:
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000/`.

## Author

**David Oyeleke**
* Email: [davidoyeleke23@gmail.com](mailto:davidoyeleke23@gmail.com)
* LinkedIn: [davidoyeleke23](https://www.linkedin.com/in/davidoyeleke23)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


