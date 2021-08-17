# Here libraries with the functions that we are going to use in this part of the project are imported.

import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage import exposure
from skimage.filters import sobel
from skimage.segmentation import watershed
import math
from skimage.morphology import reconstruction, square, disk, cube, ball



# Implementation of Perona-Malik anisotropic filtering (Aula Virtual) in order to smooth the image.
# Thanks to this implementation, original image is going to be smooth in order to get a more homogenous lung. This could help to get better result using the WaterShed segmentation.
def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,plot_flag=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                plot_flag - if True, the image will be plotted
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
 
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
 
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
 
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
 
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
 
        # initialize output array
        img = img.astype('float64')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()    
 
        for ii in range(niter):
 
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
 
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
 
                # update the image
                imgout += gamma*(NS+EW)
 
                               
        if plot_flag:
             # create the plot figure, if requested
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(img, cmap=plt.cm.gray)
            plt.title('Original image'), plt.axis('off')
            plt.subplot(122)
            plt.imshow(imgout, cmap=plt.cm.gray)
            plt.title('Filtered image (Anisotropic Diffusion)'), plt.axis('off')
 
        return imgout




# Imimposemin algorithm modifies the image in such a way that it has a minimum level of gray at the points selected, becoming local minima. This function is available in Aula Virtual. To work with this function, two images are entered as inputs: Gradient image that is going to be modified and the binary mask image with the points where we want to impose the marked minimums. The procedure to obtain both images is explained in the following funtion called WatershedExerciseP2.

'''
Python implementation of the imimposemin function in MATLAB.
Reference: https://www.mathworks.com/help/images/ref/imimposemin.html
'''

# I is the gradient/sobel image.
# BW is a binary mask.

def imimposemin(I, BW, conn=None, max_value=255):
    
    if not I.ndim in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")
    

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")
        
    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn == None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW]                 = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW]                 = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range*0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method='dilation', selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J




# This is the WaterShed algorithm created.The input parameters of this function are the image in which WaterShed Segmentation is going to be applied (img), the number of structures to be segmented (points) and a plot parameter that is going to be use depending on the neccesity to plot the final results or not. 
def WatershedExerciseP2(img, points, plot_flag= False):
    
    
# First thing this function is going to do is calculate the gradient of the image using a derivative filter such as Sobel. 
    img_eq = exposure.equalize_hist(img) # Due to the lack of contrast in our images (that avoid the presence of edges), it is decided to perform equalization of the histogram before applying sobel filter. Thanks to equalization, contrast of the image is improved. 
    image_sobel = sobel(img_eq) # Next step, is to apply a derivative filter, in our case Sobel. Thanks to previous equalization sobel is going to have a better result.
    sobel_norm = image_sobel/np.max(image_sobel) # Normalization of the sobel image in order to enhance more the contrast.
    
    
# In order to work with imimposemin function (2nd step of the algorithm), it is needed to create the two images that are going to be entered as inputs. The gradient image that has been created above this lines and the binary mask image with the points where we want to impose the marked minimums. 
# To get the image with the local minimums, you can reuse the ginput function and apply it on the gradient image. 

    mask = np.zeros((len(sobel_norm), len(sobel_norm[0]))) # Dimensions of mask must be the same that the dimensions of gradient image.
    matplotlib.use('TkAgg') # Used to be apply ginput function and select seeds.
    plt.imshow(sobel_norm, cmap ="gray")
    coord = plt.ginput(points) # Thanks to ginput function, points that represent the structures to be segmented are going to be choose by ourselves.
    for i in range(points): # A for loop is created to round and convert into integers the coordinates of the points where segmentations is going to be carried out. Besides, points selected are going to be transformed to white.
        y = int(coord[i][0].round())
        x = int(coord[i][1].round())
        mask[x][y] = 1

        
# Second step to perform with this algorithm is applied the imimposemin function that has been created previously. With this function, it is going to be impose a series of seeds where it is going to be consider that local minima are. 
    minlocal_img = imimposemin(sobel_norm, mask, conn=8, max_value=255) # Input parameters of this function are the gradient image, the binasy mask that has been previously created. Besides, a connectivity of 8 is going to be consider in order to study the 8 neighbors of the selected pixel.
    
    
# Final process is to apply Watershed function (from skimage package). This process can be done in two different ways:

# 1st way: WaterShed function is applied to the local minima image obtained previously.
    watershed_img = watershed(minlocal_img, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False)
    
# 2nd way:WaterShed function is applied to the gradient image without looking for local minima of image.
    watershed2_img = watershed(sobel_norm, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False)    
    
    if plot_flag:
        
            # create the plot figure, if requested
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(watershed_img, cmap=plt.cm.gray) # Watershed segmentation using local minima
            plt.title('Watershed mask (with local minima'), plt.axis('off')
            plt.subplot(122)
            plt.imshow(watershed2_img, cmap=plt.cm.gray) # Watershed segmentation without using local minima
            plt.title('Watershed mask (without local minima'), plt.axis('off')
    
    return img, sobel_norm, minlocal_img, watershed_img, watershed2_img

#Finally, this function returns original image, gradient image, image with corresponding local minima and  two segmentation masks (one without using imimposemin, and one using it).