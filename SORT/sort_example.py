import sort
import os
import numpy as np
import time
import json

total_time = 0.0
total_frames = 0
#Read detections file 
with open('detection-3.json') as json_file:
    json_data = json.load(json_file)
    
detections = []
for key in json_data:
    for det in json_data[key]:
        
        detection = [int(key),-1,*(det['bbox']),det['score'],-1,-1,-1]
        detections.append(detection)
#Convert to Numpy array
detections = np.asarray(detections)

    
mot_tracker = sort.Sort()
result_tracking = []
print("Processing ")
for frame in range(int(detections[:,0].max())):
    frame +=1
    #dets = seq_dets[seq_dets[:,0]==frame,2:7]
    dets = detections[detections[:,0]==frame,2:7]
    dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
    total_frames += 1

    start_time = time.time()
    trackers = mot_tracker.update(dets)
    cycle_time = time.time() - start_time
    total_time += cycle_time
    
    for d in trackers:
        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
        result_tracking.append([frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]])

print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time)) 

