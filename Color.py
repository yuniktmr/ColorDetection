import cv2
import numpy as np

feed = cv2.VideoCapture(0)
lower = np.array([33,80,40])
upper = np.array([102,255,255])
kernel = np.ones((5,5))

while True:
	ret, frame = feed.read()
	if ret is True:
		
		
		
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(img, lower, upper )
		cv2.imshow("mask", mask)
		erode = cv2.erode(mask, kernel)
		dilation = cv2.dilate(erode,kernel,iterations = 1)
		cv2.imshow("erode", erode)
		cv2.imshow("dilation", dilation)
		contours, _ = cv2.findContours(dilation.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(frame, contours, -1, (255,0,0),3)
		contours_poly = [None]*len(contours)
		boundRect = [None]*len(contours)
		for i, c in enumerate((contours)):
			x,y,w,h = cv2.boundingRect(contours[i])
			cv2.rectangle(frame, (x,y),(x+w, y+h), (255,0,0),2)
			cv2.putText(frame, str(i+1),(x, y+h),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255),2)
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFFFF == ord('q'):
			break
	else:
		continue
feed.release()
cv2.destroyAllWindows()