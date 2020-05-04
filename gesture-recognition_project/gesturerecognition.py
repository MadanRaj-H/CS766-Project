import cv2
import numpy as np

background = None

if __name__ == "__main__":
    weight_avg = 0.5

    # interested ROI coordinates
    top, right, bottom, left = 100, 700, 600, 1200

    # Load camera input
    camera_input = cv2.VideoCapture(0)

    # number of frames
    number_frames = 0

    while True:
        # current frame is captured
        rval, input_frame = camera_input.read()

        # flip the frame so that it is not the mirror view
        input_frame = cv2.flip(input_frame, 1)
        clone_frame = input_frame.copy()

        # ROI Frame
        roi = input_frame[top:bottom, right:left]

        # converting ROI to gray and blur it
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_image, (7, 7), 0)

        # To set the background and update our weight to reach the threshold
        if number_frames < 30:
            if background is None:
                background = gray_blur.copy().astype("float")
            else:
                # compute weighted average, accumulate it and update the background
                cv2.accumulateWeighted(gray_blur, background, weight_avg)
        else:
            diff = cv2.absdiff(background.astype("uint8"), gray_blur)
            kernel_square = np.ones((11, 11), np.uint8)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Perform morphological transformations in filtering out the background noise like dilasion, erosion, median
            dilation = cv2.dilate(diff, kernel_ellipse, iterations=1)
            erosion = cv2.erode(dilation, kernel_square, iterations=1)
            filtered = cv2.medianBlur(erosion, 5)
            median = filtered

            # threshold the image to get the foreground
            threshold_image = cv2.threshold(median, 25, 255, cv2.THRESH_BINARY)[1]
            contours, hierachy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            segmented = None
            if len(contours) > 0:
                # maximum contour which is the hand
                segmented = max(contours, key=cv2.contourArea)

            if segmented is not None:
                # Draw the contour
                cv2.drawContours(clone_frame, [segmented + (right, top)], -1, (0, 0, 255))
                # show the thresholded image
                cv2.imshow("Thesholded", threshold_image)
                # draw the bounding rectangle
                cv2.rectangle(clone_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                if rval:
                    print("pass it to model and get the result")

        number_frames = number_frames + 1

        # update the frame with contour and bounding rect
        cv2.imshow("Frames", clone_frame)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    camera_input.release()
    cv2.destroyAllWindows()
