import cv2

def extract_frames(video_path, num_frames=5):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // (num_frames + 1)

    for i in range(1, num_frames + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        
        ret, frame = cap.read()
        if ret:
            filename = f"sample_frame_{i}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        else:
            print(f"Error: Couldn't read frame {i * interval}")
    cap.release()

if __name__ == "__main__":
    video_path = "sample.mp4"  
    extract_frames(video_path)
