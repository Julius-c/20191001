import cv2
import glob as gb
from tqdm import tqdm

fig_dir_path = "ccmedia"
img_paths = gb.glob(fig_dir_path+"/*.png")
img_paths = [int(p.replace(fig_dir_path,"").replace("/","").replace(".png",""))for p in img_paths]
img_paths.sort()

millis_min = img_paths[0]
millis_max = img_paths[-1]
print("min #: ", millis_min)
print("max #: ", millis_max)

frame_size = (2560, 1440)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('printer.avi', fourcc, 32, frame_size)
for f in tqdm(range(millis_min,millis_max,32)):
    has_flag = False
    has_f = f
    for k in range(0, 17):
        cur_f = f + k
        if cur_f in img_paths:
            has_flag = True
            has_f = cur_f
            break
        cur_f = f - k
        if cur_f in img_paths:
            has_flag = True
            has_f = cur_f
            break
    if has_flag == False:
        has_f = f - 32; # 如果没有找到目标帧，则使用前一帧补偿
        print("?")
    if has_f in tqdm(img_paths):
        path = fig_dir_path+"/"+str(has_f)+".png"
        frame = cv2.imread(path)
        videoWriter.write(frame)
videoWriter.release()
