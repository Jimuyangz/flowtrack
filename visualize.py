import numpy as np
import cv2
import os
results = np.load('results/tracking_result.npy', allow_pickle=True)
root = '/data/OPT4/MOT17/test_flowtrack/MOT17-01-FRCNN/'
img_root = os.path.join(root,'img1')
img_n = os.listdir(img_root)
img_n.sort()
img_list = []
for i in img_n:
	img = cv2.imread(os.path.join(img_root, i))
	img = cv2.resize(img, (1920, 1024))
	img_list.append(img)

for k, track in enumerate(results):
	for b in track:
		x0 = int(b[0])
		y0 = int(b[1])
		x1 = int(b[2])
		y1 = int(b[3])
		idx = int(b[4])
		ch = int(b[5])
		im = img_list[idx]
		cv2.rectangle(im, (x0,y0), (x1,y1), (255,0,255), 2)
		cv2.putText(im,str(k)+'-'+str(ch),(x0,y0),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
		# cv2.putText(im,str(k),(x0,y0),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
		img_list[idx] = im

for k, i in enumerate(img_list):
	img_rearange = os.path.join('/data/OPT4/flowtrack/results/')
	if not os.path.exists(img_rearange):
		os.makedirs(img_rearange)
	n = os.path.join(img_rearange,str(k)+'.jpg')
	cv2.imwrite(n, i)