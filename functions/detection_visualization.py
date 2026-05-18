import cv2

def visualize_detection(path_to_video, df):
    video = cv2.VideoCapture(path_to_video)

    empty_frames_number = 35
    df["block"] = (df["frame"].diff().fillna(0) >= empty_frames_number).cumsum()

    frame_index = 0
    while True:
        ret, frame = video.read()
        if not ret: break

        subset_df = df[df["frame"] == frame_index]
        if not subset_df.empty:
            for _, row in subset_df.iterrows():
                x, y = int(row["x"]), int(row["y"])
                block = row["block"]
                cv2.circle(frame, center=(x, y), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.putText(frame, str(block), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video", frame)
        frame_index += 1

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    video.release()
    cv2.destroyAllWindows()