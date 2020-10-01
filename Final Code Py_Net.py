###Final Code First Try Only Python Edition:
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd
#region props
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import measure
import argparse
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
##Firebase Import Output:##Net1
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import datetime
from PIL import Image
import urllib.request
##Net2
from firebase import firebase

def get_image():
    # Fetch the service account key JSON file contents
    cred = credentials.Certificate("yvsr-108c1-firebase-adminsdk-xvay3-d58a5e11b9.json")  ##Add your Json Here

    # Initialize the app with a service account, granting admin privileges
    app = firebase_admin.initialize_app(cred, {
        'storageBucket': 'yvsr-108c1.appspot.com',
    }, name='storage')

    bucket = storage.bucket(app=app)
    blob = bucket.blob("pic.jpg")

    #print(blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET'))

    URL = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    #print(URL)

    with urllib.request.urlopen(URL) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())

def firebase_post(post_flag,img,cx,cy,xcor,ycor):
    from firebase import firebase
    firebase = firebase.FirebaseApplication('https://yvsr-108c1.firebaseio.com/')
    r = firebase.get('/message',None)
    
    if(post_flag == 0):
        s1 = "No Data From Console Yet/PATH May not exist"
        s2 = "No Data From Console Yet/PATH May not exist"
        s3 = "No Data From Console Yet/PATH May not exist"
        s4 = "No Data From Console Yet/PATH May not exist"
        s5 = "Img Dimensions: " + str(img.shape)
        a = firebase.put('Status','status1',s1)
        b = firebase.put('Status','status2',s2)
        c = firebase.put('Status','status3',s3)
        d = firebase.put('Status','status4',s4)
        e = firebase.put('Status','status5',s5)
    elif(post_flag == 1):####Brown
        s2 = "ST Path ("+str(cx)+","+str(cy)+")"+"xcor = "+str(xcor)+"ycor = "+str(ycor)
        b = firebase.put('Status','status2',s2)
    elif(post_flag == 2):####Green
        s3 = "NT/Green Path ("+str(cx)+","+str(cy)+")"+"xcor = "+str(xcor)+"ycor = "+str(ycor)
        c = firebase.put('Status','status3',s3)
    elif(post_flag == 3):####Gray
        s1 = "Travellable Path ("+str(cx)+","+str(cy)+")"+"xcor = "+str(xcor)+"ycor = "+str(ycor)
        a = firebase.put('Status','status1',s1)
    elif(post_flag == 4):
        s4 = "RoNI ("+str(cx)+","+str(cy)+")"+"xcor = "+str(xcor)+"ycor = "+str(ycor)
        d = firebase.put('Status','status4',s4)


def add_images(img1,img2):
    img_added = cv2.add(img1,img2)
    return(img_added)
    

def color_img_processing(img1,arrow,rhombus,square,triangle):
    img=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)####All CIP in HSV
    color=img[20,20]
    print(color)
    h,w,_=img.shape
    a=(int)(h/2)
    c1=0
    i=0
    j=0
    x1=0
    x2=0
    rols,cols,channels = img.shape
#######ALL Known Ranges
    lg1 = np.array([0, 50, 140])          #brown range
    ug1= np.array([20, 170, 255])
    lg2=np.array([30, 50, 0])            #green range
    ug2=np.array([70, 255, 255])
    lg3=np.array([0, 0, 160])            #lightgrey range
    ug3=np.array([179, 100, 240])
    lg4=np.array([0, 0, 80])            #mediumgrey range
    ug4=np.array([179,100, 160])
    lg5=np.array([0, 0, 0])            #darkgrey range
    ug5=np.array([179,100,80])


    mask1 = cv2.inRange(img, lg1, ug1)
    mask2 = cv2.inRange(img, lg2, ug2)
    mask3 = cv2.inRange(img, lg3, ug3)
    mask4 = cv2.inRange(img, lg4, ug4)
    mask5 = cv2.inRange(img, lg5, ug5)
    masked_img1 = cv2.bitwise_and(img, img, mask=mask1)
    masked_img1 = cv2.medianBlur(masked_img1,9)

    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = masked_img1[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [0,0,0]
              
    masked_img2 = cv2.bitwise_and(img, img, mask=mask2)
    masked_img2 = cv2.medianBlur(masked_img2,9)
    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = masked_img2[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [0,0,0]
    #plt.imshow(img)
    #plt.show()
    masked_img3 = cv2.bitwise_and(img, img, mask=mask3)
    masked_img3 = cv2.medianBlur(masked_img3,9)
    for i in range(0,a):
         for j in range(0,w-1):
             temp = masked_img3[i,j]
             if all(temp != [0,0,0]):    
                masked_img3[i,j] = [0,0,0]
         
    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = masked_img3[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [0,0,0]
                
    masked_img4 = cv2.bitwise_and(img, img, mask=mask4)
    masked_img4 = cv2.medianBlur(masked_img4,9)
    for i in range(0,a):
         for j in range(0,w-1):
             temp = masked_img4[i,j]
             if all(temp != [0,0,0]):    
                masked_img4[i,j] = [0,0,0]
    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = masked_img4[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [0,0,0]
    masked_img5 = cv2.bitwise_and(img, img, mask=mask5)
    masked_img5 = cv2.medianBlur(masked_img5,9)
    for i in range(0,a):
         for j in range(0,w-1):
             temp = masked_img5[i,j]
             if all(temp != [0,0,0]):    
                masked_img5[i,j] = [0,0,0]
    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = masked_img5[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [0,0,0]
    for i in range(0,h-1):
         for j in range(0,w-1):
             temp = img[i,j]
             if all(temp != [0,0,0]):    
                img[i,j] = [175,255,255]
                
    temp=cv2.bitwise_or(masked_img3,masked_img4)            
    masked_img6=cv2.bitwise_or(temp,masked_img5)

    ##Just for checking the function
##    plt.subplot(331),plt.imshow(masked_img1)
##    plt.title('1'), plt.xticks([]), plt.yticks([])
##    plt.subplot(332),plt.imshow(masked_img2)
##    plt.title('2'), plt.xticks([]), plt.yticks([])
##    plt.subplot(333),plt.imshow(masked_img3)
##    plt.title('3'), plt.xticks([]), plt.yticks([])
##    plt.subplot(334),plt.imshow(masked_img4)
##    plt.title('4'), plt.xticks([]), plt.yticks([])
##    plt.subplot(335),plt.imshow(masked_img5)
##    plt.title('5'), plt.xticks([]), plt.yticks([])
##    plt.subplot(336),plt.imshow(masked_img6)
##    plt.title('6'), plt.xticks([]), plt.yticks([])
##    plt.subplot(337),plt.imshow(img)
##    plt.title('7'), plt.xticks([]), plt.yticks([])
##
##    plt.show()

    for i in range(0,h):
         for j in range(0,w):
              if masked_img1[i,j,2]==0:                 #number of black pixels
                   c1=c1+1
    print(c1/(h*w))               
    if (c1/(h*w)<0.9):#atleast 10% detection
         print('semi travellable path exists')
         flag_brown = 1
    else:
         print('semi travellable path does not exists')
         flag_brown = 0
             
    c2=0
    for i in range(0,h):
         for j in range(0,w):
              if masked_img2[i,j,2]==0:                 #number of black pixels
                   c2=c2+1
    print(c2/(h*w))               
    if (c2/(h*w)<0.9):                                #atleast 10%  detection
        print('non travellable path exists')
        flag_green = 1
    else:
        print('non travellable path does not exists')
        flag_green = 0
    c3=0
    for i in range(0,h):
         for j in range(0,w):
              if masked_img6[i,j,2]==0:                 #number of black pixels
                   c3=c3+1
    print(c3/(h*w))               
    if (c3/(h*w)<0.9):                                #atleast 10% detection
        print('travellable path exists')
        flag_grey = 1
    else:
        print('travellable path does not exists')
        flag_grey = 0
            
    print(3-((c1+c2+c3)/(h*w)))

    ##Converting images we need to rgb
    st = cv2.cvtColor(masked_img1, cv2.COLOR_HSV2BGR)##Brown
    nt = cv2.cvtColor(masked_img2, cv2.COLOR_HSV2BGR)##Green
    tt = cv2.cvtColor(masked_img6, cv2.COLOR_HSV2BGR)##Grey
    nu = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)##No Use part

    ##Displaying the images we need:
##    plt.subplot(221),plt.imshow(st)
##    plt.title('Brown'), plt.xticks([]), plt.yticks([])
##    plt.subplot(222),plt.imshow(nt)
##    plt.title('Green'), plt.xticks([]), plt.yticks([])
##    plt.subplot(223),plt.imshow(tt)
##    plt.title('Grey'), plt.xticks([]), plt.yticks([])
##    plt.subplot(224),plt.imshow(nu)
##    plt.title('No Use Part'), plt.xticks([]), plt.yticks([])
##    plt.show()
    
    if(flag_brown == 1):
        threshb,cxb,cyb,xcorb,ycorb = contours(st)
        img_brown = imposition(threshb,triangle,cxb,cyb,xcorb,ycorb)
        post_flag = 1
        firebase_post(post_flag,img1,cxb,cyb,xcorb,ycorb)
    if(flag_green == 1):
        threshg,cxg,cyg,xcorg,ycorg = contours(nt)
        img_green = imposition(threshg,rhombus,cxg,cyg,xcorg,ycorg)
        post_flag = 2
        firebase_post(post_flag,img1,cxg,cyg,xcorg,ycorg)
    if(flag_grey == 1):
        threshgr,cxgr,cygr,xcorgr,ycorgr = contours(tt)
        img_grey = imposition(threshgr,square,cxgr,cygr,xcorgr,ycorgr)
        post_flag = 3
        firebase_post(post_flag,img1,cxgr,cygr,xcorgr,ycorgr)

    if(flag_brown == 1 and flag_green == 1 and flag_grey == 1):
        img_added = add_images(img_brown,img_green)
        img_added = add_images(img_added,img_grey)
    elif(flag_brown == 1 and flag_green == 1 and flag_grey == 0):
        img_added = add_images(img_brown,img_green)
    elif(flag_green == 1 and flag_grey == 1 and flag_brown == 0):
        img_added = add_images(img_green,img_grey)
    elif(flag_brown == 1 and flag_grey == 1 and flag_green == 0):
        img_added = add_images(img_brown,img_grey)
    elif(flag_brown == 0 and flag_green == 0 and flag_grey == 0):
        img_added = np.uint8([img.shape[0],img.shape[1]])
    elif(flag_brown == 0 and flag_green == 0 and flag_grey == 1):
        img_added = img_grey
    elif(flag_green == 0 and flag_grey == 0 and flag_brown == 1):
        img_added = img_brown
    elif(flag_brown == 0 and flag_grey == 0 and flag_green == 1):
        img_added = img_green

    threshn,cxn,cyn,xcorn,ycorn = contours(nu)
    img_na = imposition(threshn,arrow,cxn,cyn,xcorn,ycorn)
    img_added = add_images(img_added,img_na)
    post_flag = 4
    firebase_post(post_flag,img1,cxn,cyn,xcorn,ycorn)

##    plt.imshow(img_added)
##    plt.show()
    return(img_added)
    
    
        
def contours(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ##img_sarpening
    ##change1
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])

    img = cv2.filter2D(img, -1, kernel)
    #plt.imshow(img)
    # Blurring for removing the noise 
    img_blur = cv2.bilateralFilter(img, d = 7, 
                                   sigmaSpace = 75, sigmaColor =75)
    # Convert to grayscale 
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    # Apply the thresholding
    a = img_gray.max()
    thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)[1]##Possible Solution
##    _, thresh = cv2.threshold(img_gray, a/2+60, a,cv2.THRESH_BINARY_INV)
    #plt.imshow(thresh, cmap = 'gray')
    #plt.show()
    thresh = cv2.bitwise_not(thresh)
    #plt.imshow(thresh, cmap = 'gray')
    #plt.show()
    # Find the contour of the figure 
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    # Draw the contour 
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx = -1, 
                             color = (255, 0, 0), thickness = 2)
    #plt.imshow(img_copy)
    #plt.show()
    # The first order of the contours
    c_0 = contours[0]
    # image moment
    M = cv2.moments(c_0)
    #print(M.keys())

    # The area of contours 
##    print("1st Contour Area : ", cv2.contourArea(contours[0])) # 37544.5
##    print("2nd Contour Area : ", cv2.contourArea(contours[1])) # 75.0
##    print("3rd Contour Area : ", cv2.contourArea(contours[2])) # 54.0

    # The arc length of contours 
    print(cv2.arcLength(contours[0], closed = True))      # 2473.3190
    print(cv2.arcLength(contours[0], closed = False))     # 2472.3190

    # The centroid point
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print(cx,cy)
    # The extreme points
    l_m = tuple(c_0[c_0[:, :, 0].argmin()][0])
    r_m = tuple(c_0[c_0[:, :, 0].argmax()][0])
    t_m = tuple(c_0[c_0[:, :, 1].argmin()][0])
    b_m = tuple(c_0[c_0[:, :, 1].argmax()][0])
    pst = [l_m, r_m, t_m, b_m]
    xcor = [p[0] for p in pst]
    ycor = [p[1] for p in pst]

    print("l_m = ",l_m) #left
    print("r_m = ",r_m) #right
    print("t_m = ",t_m) #top
    print("b_m = ",b_m) #bottom
    print("pst = ",pst) #all Extremes
    print("xcor = ",xcor) #x-co-ordinates of all
    print("ycor = ",ycor) #y-co-ordinates of all

##    #Plot the points##Syntax##image = cv.circle(image, centerOfCircle, radius, color, thickness)
##    plt.figure(figsize = (10, 16))
##    plt.subplot(1, 3, 1)
##    plt.imshow(thresh, cmap = 'gray')
##    plt.scatter([cx], [cy], c = 'b', s = 50)#centroid
##    plt.subplot(1, 3, 2)
##    plt.imshow(thresh, cmap = 'gray')
##    plt.scatter(xcor, ycor, c = 'b', s = 50)#All x and y co-ordinates i.e. all extremes
    img = cv2.circle(img, (xcor[0],ycor[0]), 2, (255, 0, 255), 11)
    img = cv2.circle(img, (xcor[1],ycor[1]), 2, (255, 0, 255), 10)
    img = cv2.circle(img, (xcor[2],ycor[2]), 2, (255, 0, 255), 10)
    img = cv2.circle(img, (xcor[3],ycor[3]), 2, (255, 0, 255), 10)
##    plt.subplot(1, 3, 3)
##    plt.imshow(img, cmap = 'gray')
##    plt.show()
    return(img,cx,cy,xcor,ycor)

##    Using only the BW part for the imposition
##    thresh = cv2.circle(thresh, (cx,cy), 2, (0, 0, 255), 10)
##    thresh = cv2.circle(thresh, (xcor[0],ycor[0]), 2, (255, 0, 255), 11)
##    thresh = cv2.circle(thresh, (xcor[1],ycor[1]), 2, (255, 0, 255), 10)
##    thresh = cv2.circle(thresh, (xcor[2],ycor[2]), 2, (255, 0, 255), 10)
##    thresh = cv2.circle(thresh, (xcor[3],ycor[3]), 2, (255, 0, 255), 10)
##    plt.subplot(1, 3, 3)
##    plt.imshow(thresh, cmap = 'gray')
##    plt.show()
##    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
##    return(thresh,cx,cy,xcor,ycor)

def imposition(img1,img2,cx,cy,xcor,ycor):

    ydif = ycor[1]-ycor[0]
    ydif = int(ydif/2)
    cx = int(cx)
    cy = int(cy)
    #Resizing the Image that we want to impose
    if(img1.size < 250000):
        scale_percent = 25
    else:
        scale_percent = 60 # percent of original size
    width = int(img2.shape[1] * scale_percent / 100)
    height = int(img2.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

    # I create an ROI
    rows,cols,channels = img2.shape
    roi = img1[ycor[1]-ydif:ycor[1]-ydif+rows,  xcor[1]-ydif:xcor[1]-ydif+cols]
    
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    try:
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        flag = 0
    except:
        try:
            rows,cols,channels = img2.shape
            roi = img1[cx:cx+rows,  cy:cy+cols]
            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            flag = 1

        except:
            try:
                ydif = 0
                rows,cols,channels = img2.shape
                roi = img1[ycor[0]+ydif:ycor[0]+ydif+rows,  xcor[0]+ydif:xcor[0]+ydif+cols]
                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                flag = 2
            except:
                try:
                    ydif = 0
                    rows,cols,channels = img2.shape
                    roi = img1[ycor[2]+ydif:ycor[2]+ydif+rows,  xcor[2]+ydif:xcor[2]+ydif+cols]
                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    flag = 3
                except:
                    ydif = ycor[2]-ycor[3]
                    ydif = int(ydif/2)
                    rows,cols,channels = img2.shape
                    roi = img1[ycor[3]-ydif:ycor[3]-ydif+rows,  xcor[3]-ydif:xcor[3]-ydif+cols]
                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    flag = 4

    #img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    
    if(flag == 0):
        img1[ycor[1]-ydif:ycor[1]-ydif+rows,  xcor[1]-ydif:xcor[1]-ydif+cols] = dst
    elif(flag == 1):
        img1[cx:cx+rows, cy:cy+cols] = dst
    elif(flag == 2):
        img1[ycor[0]+ydif:ycor[0]+ydif+rows,  xcor[0]+ydif:xcor[0]+ydif+cols] = dst
    elif(flag == 3):
        img1[ycor[2]+ydif:ycor[2]+ydif+rows,  xcor[2]+ydif:xcor[2]+ydif+cols] = dst
    elif(flag == 4):
        img1[ycor[3]-ydif:ycor[3]-ydif+rows,  xcor[3]-ydif:xcor[3]-ydif+cols] = dst
##    plt.imshow(img1)
##    plt.show()

    return(img1)

def main():
    get_image()
    img = cv2.imread('temp.jpg')
    #broad2-only st#broad-4(Defect)#road2-2#road3-only T
    ##road4-onlt ST###Use road15
    #reading all extra images##Big images take a minute approx
    post_flag = 0
    a = 0
    firebase_post(post_flag,img,a,a,a,a)
    arrow = cv2.imread('Arrow.jpg')#Above Conceivable Height
    rhombus = cv2.imread('Rhombus.jpg')#Untravellable Path/Rough Terrain
    square = cv2.imread('Square.jpg')#Travellable Path
    triangle = cv2.imread('Triangle.jpg')#Semi-travellable Path
    legend = cv2.imread('legend.jpg')#Legend
    output = color_img_processing(img,arrow,rhombus,square,triangle)
    #Plot the points##Syntax##image = cv.circle(image, centerOfCircle, radius, color, thickness)
    plt.figure(figsize = (10, 16))
    plt.subplot(1, 3, 1)
    plt.title('Org Img'), plt.xticks([]), plt.yticks([]),plt.imshow(img, cmap = 'gray')
    plt.subplot(1, 3, 2)
    plt.title('Map'), plt.xticks([]), plt.yticks([]),plt.imshow(output, cmap = 'gray')
    plt.subplot(1, 3, 3)
    plt.title('Legend'), plt.xticks([]), plt.yticks([]),plt.imshow(legend, cmap = 'gray')
    plt.show()

if __name__ == '__main__':
    main()
