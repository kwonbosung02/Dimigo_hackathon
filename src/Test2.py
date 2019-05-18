import cv2
import numpy as np

img1 = cv2.imread("jelly.jpg", cv2.IMREAD_GRAYSCALE)

cam = cv2.VideoCapture(0)
cam.set(3,960)
cam.set(4,480)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)





if __name__ == "__main__":
    print("OpencvImgTest")

    while(1):
        frame, img = cam.read()
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(frame, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        matching_result = cv2.drawMatches(img1, kp1, frame, kp2, matches[:10], None, flags=2)

        cv2.imshow("Img1", img1)        
        cv2.imshow("Img2", frame)
        cv2.imshow("Matching result", matching_result)
        print(len(matches))
        if len(matches) > 130 :
            print("mango_Jelly_detected")
        else :
            print("nothing_detected")

        k= cv2.waitKey(10) & 0xff
        if k==27:
            cam.release()
            cv2.destroyAllWindows()

            break

cv2.waitKey(0)

cv2.destroyAllWindows()