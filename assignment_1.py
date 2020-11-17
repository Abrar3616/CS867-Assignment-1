import os
import cv2
import glob
import time
import random
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputfolder ="/home/abrar/images"

def reading():
    for image in glob.glob(inputfolder + "/*"): # find image from folder
        img = cv2.imread(image) # read image
        imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to grayscale
        img3channel = cv2.cvtColor(imggray, cv2.COLOR_GRAY2RGB) # convert to 3 channel again for stacking

        out =  np.hstack((img,img3channel)) # stack the images
        plt.imshow(out), plt.show() # show images

def rgbExclusion():
    colour = input("Enter Colour to Exclude:") # take input form user to determine which colour to exclude
    for image in glob.glob(inputfolder + "/05.jpeg"): # find image from folder
        img = cv2.imread(image) # read image
        if(colour == "red" or colour == "r"):
            img[:,:,0] = 0 # remove red colour
            plt.imshow(img), plt.show()

        if(colour == "green" or colour == "g"):
            img[:,:,1] = 0 # remove green colour
            plt.imshow(img), plt.show()

        if(colour == "blue" or colour == "b"):
            img[:,:,2] = 0 # remove blue colour
            plt.imshow(img), plt.show()

def histogram():
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image, 0) # read image in grayscale
        hist = cv2.calcHist([img],[0],None,[256],[0,256]) # find histogram of image

        eq = cv2.equalizeHist(img) # equalize histogram of image
        histeq = cv2.calcHist([eq],[0],None,[256],[0,256]) # display histogram of equalized image

        # plot images and histograms
        fig = plt.figure()
        gs = fig.add_gridspec(2, 30)

        fig.add_subplot(gs[0,0:10])
        fig.subplots_adjust(wspace=1)
        plt.imshow(img, cmap="gray", aspect="auto")
        fig.add_subplot(gs[0,13:-1])
        fig.subplots_adjust(wspace=1)
        plt.plot(hist)

        fig.add_subplot(gs[1,0:10])
        fig.subplots_adjust(wspace=1)
        plt.imshow(eq, cmap="gray", aspect="auto")
        fig.add_subplot(gs[1,13:-1])
        fig.subplots_adjust(wspace=1)
        plt.plot(histeq)

        plt.show()

def conv():
    path = input("Enter Image Path('def' for default image): ") # take input of image path from user
    if(path == "def"):
        path = inputfolder + "/04.jpg" # set default image for testing
    img = cv2.imread(path, 0) # read image as grayscale

    type = input("Enter Type of Convolution('sharpen' or 'smooth'): ") # take input of desired kernel from user
    if(type == "sharpen"):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif(type == "smooth"):
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9

    # do convolution
    a, b = kernel.shape
    y, x = img.shape
    y = y-a+1
    x = x-b+1
    out = np.zeros((y,x)) # make an output image to paste output of convolution
    for i in range(y):
        for j in range(x):
            out[i][j] = np.sum(img[i:i+a, j:j+b]*kernel)

    # plot images
    fig = plt.figure()
    fig.add_subplot(121)
    fig.suptitle("Image is " + type + "ed")
    plt.imshow(img, cmap="gray")
    fig.add_subplot(122)
    plt.imshow(out, cmap="gray")
    plt.show()

def box():
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image)
        kernel = kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])/25 # box filter kernel
        box = cv2.filter2D(img, -1, kernel) # apply filter

        # plot images
        plt.subplot(121), plt.imshow(img), plt.title("Original")
        plt.subplot(122), plt.imshow(box), plt.title("Box Filtered")
        plt.show()

def gaus():
    sigma = float(input("Enter sigma value for filter= ")) # take input of sigma value from user
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image)
        gaus = cv2.GaussianBlur(img, (3,3), sigma) # apply gaussian filter

        # plot images
        plt.subplot(121), plt.imshow(img), plt.title("Original")
        plt.subplot(122), plt.imshow(gaus), plt.title("Gaussian Filtered")
        plt.show()

def gausnoise():
    sigma = float(input("Enter sigma value for noise= ")) # take sigma value from user
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image)
        r, c, ch = img.shape
        noisyimg = np.zeros(img.shape, np.float32)
        gaussian = np.random.normal(0, sigma, (r, c, 3))
        gaussian = gaussian.reshape(r,c,3)
        noisyimg = img + gaussian # add gaussian noise
        cv2.normalize(noisyimg, noisyimg, 0, 255, cv2.NORM_MINMAX, dtype=-1) # normalize the image
        noisyimg = noisyimg.astype(np.uint8)
        gaus = cv2.GaussianBlur(noisyimg, (3,3), 0) # apply gaussian filter

        # plot images
        plt.subplot(131), plt.imshow(img), plt.title("Original")
        plt.subplot(132), plt.imshow(noisyimg), plt.title("Gaussian Noise")
        plt.subplot(133), plt.imshow(gaus), plt.title("Gaussian Filtered")
        plt.show()

def saltandpepper():
    prob = float(input("Enter probability of S&P Noise= ")) # take input of probability of salt and pepper noise from user
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image)
        noisyimg = np.copy(img)

        # define salt and pepper
        if len(img.shape) == 2:
            pepper = 0
            salt = 255
        else:
            colorspace = img.shape[2]
        if colorspace == 3:  # RGB
            pepper = np.array([0, 0, 0], dtype='uint8')
            salt = np.array([255, 255, 255], dtype='uint8')

        # add salt and pepper
        rnd = np.random.random(img.shape[:2])
        noisyimg[rnd < (prob/2)] = pepper
        noisyimg[rnd > 1 - (prob/2)] = salt

        sandp = cv2.medianBlur(noisyimg,3) # salt and pepper filter

        # plot images
        plt.subplot(131), plt.imshow(img), plt.title("Original")
        plt.subplot(132), plt.imshow(noisyimg), plt.title("S & P Noise")
        plt.subplot(133), plt.imshow(sandp), plt.title("S & P Filtered")
        plt.show()

def mesh():
    sigma = float(input("Enter sigma value for mesh plots= "))
    img = np.zeros((101, 101))
    img[50,50] = 1
    X, Y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # for gaussian mesh
    imggaus = ndi.gaussian_filter(img,sigma,order=[0,0],output=np.float64, mode='nearest')

    # for first order derivative of gaussian mesh
    imggausder = ndi.gaussian_filter(img,sigma,order=[1,0],output=np.float64, mode='nearest')

    # for laplacian of gaussian mesh
    imglap = ndi.gaussian_laplace(img,sigma,output=np.float64, mode='nearest')

    # make 3D plot
    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    ax.plot_surface(X, Y, imggaus,cmap='viridis',linewidth=0)
    ax.set_title("Gaussian Mesh")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    fig2 = plt.figure()
    bx = fig2.gca(projection='3d')
    bx.plot_surface(X, Y, imggausder, cmap="viridis",linewidth=0)
    bx.set_title("First Order X Derivative of Gaussian Mesh")
    bx.set_xlabel('X axis')
    bx.set_ylabel('Y axis')
    bx.set_zlabel('Z axis')

    fig3 = plt.figure()
    cx = fig3.gca(projection='3d')
    cx.plot_surface(X, Y, imglap, cmap="viridis",linewidth=0)
    cx.set_title("Laplacian of Gaussian Mesh")
    cx.set_xlabel('X axis')
    cx.set_ylabel('Y axis')
    cx.set_zlabel('Z axis')
    plt.show()




def sobel():
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image, 0)
        y, x = img.shape
        gradmag = np.zeros((y,x))
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # derivative in x direction
        abs_sobelx = np.absolute(sobelx) # absolute of gradient in x
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) # derivative in y direction
        abs_sobely = np.absolute(sobely) # absolute of gradient in y
        cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0, gradmag); # gradient magnitude

        # plot images
        plt.subplot(221), plt.imshow(img, cmap="gray"), plt.title("Original")
        plt.subplot(222), plt.imshow(gradmag, cmap="gray"), plt.title("Gradient Mag")
        plt.subplot(223), plt.subplots_adjust(hspace=0.4), plt.imshow(sobelx, cmap="gray"), plt.title("Gradient X")
        plt.subplot(224), plt.subplots_adjust(hspace=0.4), plt.imshow(sobely, cmap="gray"), plt.title("Gradient Y")
        plt.show()

def LoG():
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image, 0)
        y, x = img.shape
        gradmag = np.zeros((y,x))
        LoG = cv2.Laplacian(img,cv2.CV_64F, ksize=5)
        abs_log = np.absolute(LoG)
        cv2.addWeighted(abs_log, 0.5, gradmag, 0.5, 0, gradmag); # gradient magnitude

        # plot images
        plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title("Original")
        plt.subplot(132), plt.imshow(LoG, cmap="gray"), plt.title("Lap of Gaus")
        plt.subplot(133), plt.imshow(gradmag, cmap="gray"), plt.title("Grad of LoG")
        plt.show()

def canny():
    for image in glob.glob(inputfolder + "/*.jpg"): # read images
        img = cv2.imread(image, 0)
        imgsm = cv2.GaussianBlur(img, (3,3), 0) # apply gaussian filter
        y = img.shape[0]
        x = img.shape[1]
        canny = np.zeros((y, x, 3))
        canny = cv2.Canny(imgsm, None, 10, 30, 3) # detect canny edges

        # plot images
        plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("Original")
        plt.subplot(122), plt.imshow(canny, cmap="gray"), plt.title("Canny Edges")
        plt.show()

def cannyvid(): # Due to Codec mismatch video is not being saved and i have tried many codecs but i cant seem to find the correct combination
    t0 = time.time() # note start time
    cap = cv2.VideoCapture(0) # receive live stream
    fps = 30.0
    #capture_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D') # codec for output video
    out = cv2.VideoWriter('/home/abrar/CannyVid.avi', fourcc, fps, (640,480)) #define output video object

    while(True): # while we are receiveing stream
        ret, frame = cap.read() # capture frame-by-frame
        t1 = time.time() # note time
        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # make video grayscale
            frame = cv2.GaussianBlur(fgray, (3,3), 0) # apply gaussian filter
            y = frame.shape[0]
            x = frame.shape[1]
            cframe = np.zeros((y,x, 3))
            cframe = cv2.Canny(frame, None, 100, 300, 3) # canny edge detection on each frame

            out.write(cframe) # write frame to video
            plt.imshow(cframe, cmap="gray"), plt.show()
            num_seconds = t1 - t0 # difference in time
            if num_seconds > 10: # record 10 seconds of video
                break
            #if cv2.waitKey(0): # quit on command
            #    break
        else:
            break
    # close stream and saving video
    cap.release()
    out.release()

    show = cv2.VideoCapture('/home/abrar/CannyVid.avi') # open video for output

    while(show.isOpened()):
        cret, vframe = show.read() # read saved video

        if cret == True:
            cv2.imshow('frame',vframe) # show video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # close everything
    show.release()
    cv2.destroyAllWindows()
