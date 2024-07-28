import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Capture image from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 's' to capture an image.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save the captured image
        cv2.imwrite('captured_image.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()

# Load the captured image5

raw_image = Image.open('captured_image.jpg').convert('RGB')

# Get the number of questions from the user
num_questions = int(input("How many questions would you like to ask? "))

for i in range(num_questions):
    # Get the question from the user
    question = input(f"Please enter question {i + 1}: ")

    # Process the image and the question
    inputs = processor(raw_image, question, return_tensors="pt")

    # Generate the answer
    out = model.generate(**inputs)

    # Print the answer
    answer = processor.decode(out[0], skip_special_tokens=True)
    print(f"Answer to question {i + 1}: {answer}")
