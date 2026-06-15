import numpy as np

def correct_detection(frame_list):
    isolated_points = []
    for i in range(1, len(frame_list)-1):
        x_before, y_before = frame_list[i-1][0], frame_list[i-1][1]
        x, y = frame_list[i][0], frame_list[i][1]
        x_after, y_after = frame_list[i+1][0], frame_list[i+1][1]
        
        distance1 = np.sqrt((x - x_before)**2 + (y - y_before)**2)
        distance2 = np.sqrt((x - x_after)**2 + (y - y_after)**2)

        if (distance1 > 5 and distance2 > 5) or abs(distance1 - distance2) > 3:
            isolated_points.append(i)

    clean_list = [(i, p) for i, p in enumerate(frame_list) if i not in isolated_points]
    return clean_list
