import requests

# Endpoint
url = "http://127.0.0.1:5000/detect_license_plate"

# Loop to send 6 placeholder images
for i in range(6):
    image_filename = f"sample_frame_{i+1}.png"  # Adjust this to point to your actual images
    with open(image_filename, 'rb') as image_file:
        response = requests.post(url, files={"image": image_file})
    
    print(f"Sent image {i+1}. Server response:", response.json())
