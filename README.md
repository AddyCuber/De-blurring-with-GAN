# Deblurring with GAN

A web application that uses a Generative Adversarial Network (GAN) to deblur images. Built with Flask and TensorFlow.

## Features

- Upload and process blurred images
- Real-time image deblurring using GAN
- Modern, responsive web interface
- Support for various image sizes while maintaining aspect ratio

## Setup

1. Clone the repository:
```bash
git clone https://github.com/AddyCuber/De-blurring-with-GAN.git
cd De-blurring-with-GAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained model:
- The model file (`deblur_generator.h5`) is not included in the repository due to size limitations
- Download it from the [Releases page](https://github.com/AddyCuber/De-blurring-with-GAN/releases/latest)
- Place the downloaded `deblur_generator.h5` file in the root directory of the project

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
De-blurring-with-GAN/
├── app.py              # Flask application
├── deblur_generator.h5 # Trained model (download from Releases)
├── templates/          # HTML templates
│   ├── index.html     # Upload page
│   └── result.html    # Results page
├── static/            # Static files
│   └── uploads/       # Uploaded images
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Technologies Used

- Python
- Flask
- TensorFlow
- HTML/CSS
- JavaScript

## Model Information

The `deblur_generator.h5` file contains a trained GAN generator model that can deblur images. The model:
- Input: Blurred RGB images (any size, will be resized internally)
- Output: Deblurred images maintaining the original aspect ratio
- Architecture: Based on a GAN with custom reflection padding
- File size: ~45MB

## License

This project is licensed under the MIT License - see the LICENSE file for details. 