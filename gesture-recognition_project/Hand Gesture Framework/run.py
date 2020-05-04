#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from flask import Flask, render_template, Response,jsonify
from imutils import face_utils
import cv2, imutils, dlib, numpy as np, os, tensorflow as tf, sys
import struct, collections
import gestureframework as plib

app = Flask(__name__)

status_text='';
detector = dlib.get_frontal_face_detector()
pl=''
probables = ''
gls=[]
feedback_decisions=[]
blink=0
indexpred = 0;
predictor = dlib.shape_predictor('models/face-detector.dat')
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])


@app.route('/')
def index():
	return render_template('home.html')

def image_resize(image):
	r = 100.0 / image.shape[1]
	dim = (100, int(image.shape[0] * r))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized
	
def get_frame():
	camera=cv2.VideoCapture(0)
	
	COUNTER = 0
	EYE_AR_THRESH = 0.22#0.2
	EYE_AR_CONSEC_FRAMES = 10
	classification_flag=0
	feedback_flag=0
	classify_count=0
	fr=1
	first_letter=[]
	first_probability=[]
	second_letter=[]
	second_probability=[]
	firstdecision='na'
	seconddecision='na'
	feedback_count=0
	predicted_letter=''
	predictions_intotal = []
	letters = []

	while True:
		grabbed, frame = camera.read()
		global status_text
		status_text='Video Capture starts.'
		size = frame.shape
		h=size[0]
		w=size[1]
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		focal_length = size[1]
		center = (size[1]/2, size[0]/2)
		camera_matrix = np.array(
								 [[focal_length, 0, center[0]],
								 [0, focal_length, center[1]],
								 [0, 0, 1]], dtype = "double"
								 )
		if len(rects) != 0:
			status_text='Face Detected!'
			maxArea = 0
			maxRect = None
			for rect in rects:
				if rect.area() > maxArea:
					maxArea = rect.area()
					maxRect = [rect.left(),rect.top(),rect.right(),rect.bottom()]	
			rect = dlib.rectangle(*maxRect)
			shape = predictor(gray, rect)
			global blink
			if blink==0:
				left_eye,right_eye,ear=plib.eyes_detection(shape)
				status_text='Eyes Detected!'
				blink = 1;
			elif blink==1:
				landmarks = plib.dlibLandmarksToPoints(shape)
				landmarks = np.array(landmarks)
				#for (x, y) in landmarks:
					#cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
				if (classification_flag==0):
					ix = landmarks[32][0]
					fx = landmarks[34][0]
					iy = landmarks[29][1]
					fy = landmarks[30][1]
					tempimg = frame[iy:fy,ix:fx,:]
					status_text='Nose skin detection!'
					meanimg = np.uint8([[cv2.mean(tempimg)[:3]]])
					skinRegionycb,skinycb = plib.findSkinYCB(meanimg, frame)
			
					maskedskinycb = plib.applyMask(skinycb, landmarks)
					#cv2.putText(skinycb, "YCrCb", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
				
					x1=int(landmarks[0][0])
					#cv2.circle(frame, (400,200),10, (255,0,0),-1)
					newframe=frame[:,0:x1]
					maskedframe=maskedskinycb[:,0:x1]
					handregion=skinRegionycb[:,0:x1]
			
					drawing,xh,yh,wh,hh=plib.handconvex(handregion,newframe,maxArea)

					if ((xh!=0) and (yh!=0) and (wh!=0) and (hh!=0)):
						status_text='Hand Detected..'
						crphand=maskedframe[yh:yh+hh,xh:xh+wh]
						r = 90.0 / crphand.shape[1]
						dim = (90, int(crphand.shape[0] * r))
						resized = crphand
			
						if classify_count<20:
							if (fr%2)==0:
								resized = cv2.resize(resized, (224,224))
								np_image_data = np.asarray(resized)
								np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
								np_final = np.expand_dims(np_image_data,axis=0)				
								predictions=plib.handclassify(np_final)
								letter,confidence=predictions[0]
								first_letter.append(letter)
								first_probability.append(confidence)
								letter,confidence=predictions[1]
								second_letter.append(letter)
								second_probability.append(confidence)
								classify_count=classify_count+1
								for i in predictions:
									predictions_intotal.append(i)
						else:
							status_text='Hand Gesture Recognized'
							classification_flag=1
							letters_counter=collections.Counter(first_letter)
							#print(letters_counter)
							letters=sorted(letters_counter.items(), key=lambda x: x[1])
							predicted_letter,freq=letters[len(letters)-1]
							global indexpred
							indexpred = len(letters)-1
							print("madan. " + str(indexpred));
							global pl
							pl='Predicted letter : '+predicted_letter
							global probables
							#probables = "Probable letters with confidence : {}".format('\n '.join(str(v) for v in reversed(letters)));
							#print('Predictions'+ str(predictions))
							probables = "Probable letters with confidence : {}".format(str(letters))
				else: 
					status='Feedback Mode..'
					global feedback_decisions
					if feedback_count<20 or len(feedback_decisions) == 0:
						feedback_count=feedback_count+1
					else:
						feedback = feedback_decisions[0]
						if feedback == 'inprobables':
							indexpred-=1;
							print("madan. " + str(indexpred));
							if (indexpred >= 0):
								predicted_letter,freq=letters[indexpred]
								#print('Next Predicted letter: '+predicted_letter)
								pl='Predicted letter : '+predicted_letter
							else:
								firstdecision='na'
								seconddecision='na'
								classification_flag=0
								classify_count=0
								pl=''
								probables = ''
								first_letter=[]
								second_letter=[]
						elif feedback == 'yes':
							gls.append(predicted_letter)
							pl='Predicted letter : '+predicted_letter
							pl=''
							probables = ''
							firstdecision='na'
							classification_flag=0
							classify_count=0
							first_letter=[]
							second_letter=[]
						elif feedback == 'no':
							firstdecision='na'
							seconddecision='na'
							classification_flag=0
							classify_count=0
							pl=''
							probables = ''
							first_letter=[]
							second_letter=[]
						else:
							print("nothing")
						feedback_decisions=[]
						feedback_count=0
						if (classification_flag==0):
							indexpred = 0
		else:
			status_text='No Faces Detected..'
			
		fr=fr+1
		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()
		yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
	del(camera)

@app.route('/status')
def status():
	status_result = {'status': status_text, 'predicted_letter': pl,'probables': probables}
	return jsonify(status_result)
@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/yes')
def yesmessage():
	global feedback_decisions
	feedback_decisions.append('yes')
	global blink
	blink = 1
	return "success"
@app.route('/inprobables')
def inprobablesmessage():
	global feedback_decisions
	feedback_decisions.append('inprobables')
	global blink
	blink = 1
	return "success"
@app.route('/no')
def nomessage():
	global feedback_decisions
	feedback_decisions.append('no')
	global blink
	blink = 1
	return "success"
if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
