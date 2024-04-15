import cv2

def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video file")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

video_path = "D:\\image\\mobile phone pic\\VID_20220902_092211(0).mp4"
play_video(video_path)
