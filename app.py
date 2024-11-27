import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model

# Load the pre-trained model for Gujarati numeral recognition
model = load_model('gujarati_numeral_recognition_model.keras', compile=False)

class DigitRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window title and size
        self.setWindowTitle("Gujarati Numeral Recognition App")
        self.setFixedSize(1000, 1000)

        # Create a vertical layout to arrange widgets
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add a title label at the top of the app
        self.title_label = QLabel("Gujarati Numeral Recognition")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; text-align: center;")
        self.title_label.setFixedHeight(50)
        self.layout.addWidget(self.title_label)

        # Add a label to display the selected image
        self.image_label = QLabel()
        self.image_label.setStyleSheet("border: 1px solid black;  border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(650)
        self.layout.addWidget(self.image_label)

        # Create a horizontal layout for buttons
        self.button_layout = QHBoxLayout()

        # Add an upload button for selecting an image
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet("font-size: 18px; background-color: green; color: white; border-radius: 10px;")
        self.upload_button.setFixedHeight(50)
        self.upload_button.clicked.connect(self.upload_image)
        self.button_layout.addWidget(self.upload_button)

        # Add a download button for saving the annotated image
        self.download_button = QPushButton("Download Annotated Image")
        self.download_button.setStyleSheet("font-size: 18px; background-color: lightblue; color: white; border-radius: 10px;")
        self.download_button.setFixedHeight(50)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_image)
        self.button_layout.addWidget(self.download_button)

        # Add the horizontal button layout to the main layout
        self.layout.addLayout(self.button_layout)

        # Add a text area to display the recognized digits
        self.digits_text_area = QTextEdit()
        self.digits_text_area.setReadOnly(True)
        self.digits_text_area.setStyleSheet("font-size: 18px; border: 1px solid black; background-color: #f0f0f0; border-radius: 10px;")
        self.digits_text_area.setFixedHeight(150)
        self.layout.addWidget(self.digits_text_area)

        # Initialize variables
        self.annotated_image = None
        self.recognized_digits = []

    def upload_image(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "test_images", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        # Enable the download button and change its color
        self.download_button.setEnabled(True)
        self.download_button.setStyleSheet("font-size: 18px; background-color: blue; color: white; border-radius: 10px;")

        # Read the image from the selected file
        image = cv2.imread(file_path)

        # Get the original dimensions of the image
        original_height, original_width = image.shape[:2]

        # Check if resizing is needed
        if original_width > 100 and original_height > 100:
            # Scale and resize the image to the image window layout preserving the aspect ratio
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            width_ratio = label_width / original_width
            height_ratio = label_height / original_height

            scaling_factor = min(width_ratio, height_ratio)

            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)

            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # If the image is smaller than 100x100, use the original image
            resized_image = image

        # Convert the image to grayscale
        grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Preprocess image for noise removal and optical character recognition
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clear previous recognized digits
        self.recognized_digits = []

        # Iterate over all the detected contours
        for c in contours:
            # Get bounding box coordinates for each contour
            x, y, w, h = cv2.boundingRect(c)

            if cv2.contourArea(c) < 50:  # Skip small contours (noise)
                continue

            # Crop the region of interest (ROI) for the digit
            digit = thresh[y:y + h, x:x + w]

            # Resize the digit to 18x18 pixels and pad it to 28x28 pixels
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            # Reshape the padded digit to the format expected by the model (1x28x28x1) and normalize
            roi_resized = np.reshape(padded_digit, (1, 28, 28, 1)) / 255.0

            # Predict the digit using the pre-trained model
            prediction = model.predict(roi_resized, verbose=0)

            # Get the predicted label (digit)
            digit_label = np.argmax(prediction)

            # Append the recognized digit and its coordinates to the list
            self.recognized_digits.append((digit_label, x, y))

            # Annotate the image with the predicted label
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(resized_image, f"{digit_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Sort the recognized digits by their position (top-left to bottom-right)
        self.recognized_digits.sort(key=lambda d: (d[2], d[1]))

        # Update the recognized digits text area
        self.update_digits_text()

        # Store the annotated image for downloading
        self.annotated_image = resized_image

        # Display the processed image with annotations
        self.display_image(resized_image)

        # Enable the download button
        self.download_button.setEnabled(True)


    def update_digits_text(self):
        # Group digits into rows based on their y-coordinate
        sorted_digits = sorted(self.recognized_digits, key=lambda d: (d[2], d[1]))
        rows = []
        current_row = []
        row_y_threshold = 20  # Adjust threshold for grouping digits in a row

        for i, (digit, x, y) in enumerate(sorted_digits):
            if i == 0 or abs(y - current_row[-1][2]) < row_y_threshold:
                current_row.append((digit, x, y))
            else:
                rows.append(current_row)
                current_row = [(digit, x, y)]
        
        if current_row:  # Append the last row
            rows.append(current_row)

        # Format the digits into rows and update the text area
        result = []
        for row in rows:
            row_digits = [str(d[0]) for d in sorted(row, key=lambda d: d[1])]
            result.append(" ".join(row_digits))

        self.digits_text_area.setText("\n".join(result))

    def display_image(self, image):
        # Convert the image to RGB format for display
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the height, width, and channels of the image
        h, w, ch = img_rgb.shape

        # Calculate the number of bytes per line (required for QImage)
        bytes_per_line = ch * w

        # Create a QImage from the image data
        q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit the image label, preserving the aspect ratio of the image
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Display the scaled pixmap on the label
        self.image_label.setPixmap(scaled_pixmap)

    def download_image(self):
        # Open a file dialog to save the annotated image
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotated Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            cv2.imwrite(file_path, self.annotated_image)


if __name__ == '__main__':
    # Initialize the application
    app = QApplication(sys.argv)

    # Create the main window for the app
    window = DigitRecognitionApp()
    window.show()

    # Start the application
    sys.exit(app.exec_())
