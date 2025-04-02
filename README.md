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
- You can download it from [here](link-to-model-file)

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
De-blurring-with-GAN/
├── app.py              # Flask application
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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 