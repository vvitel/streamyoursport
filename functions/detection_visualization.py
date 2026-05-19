import cv2

def visualize_detection(path_to_video, data):
    video = cv2.VideoCapture(path_to_video)
    X, Y, FRAME, BLOCK  = 0, 1, 5, -1

    frame_index = 0
    while True:
        ret, frame = video.read()
        if not ret: break

        subset = data[data[:, FRAME] == frame_index]
        if subset.size > 0:
            for row in subset:
                x, y = int(row[X]), int(row[Y])
                block = row[BLOCK]
                cv2.circle(frame, center=(x, y), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame, str(block), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video", frame)
        frame_index += 1

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    video.release()
    cv2.destroyAllWindows()