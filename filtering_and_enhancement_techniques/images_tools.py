#Here we import libraries with the functions that we are going to use in this project.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
import scipy.ndimage.filters as filters
from skimage.filters import sobel

##Noise

# We create a function to add noise, in this function we choose the image we want to add noise to, the type of noise we want to add and the intensity of the noise. Besides, we decide if we want to plot the resulting noisy image or not.

def noisy_image(path,noise, intensity, plot=True):
    
    
    img = mpimg.imread(path) #First, we read the image.
    img = rgb2gray(img) # Here, we convert the three true colors of image (rgb) to the gray scale image.
    
    #This is gaussian noise.
    if noise == "gaussian":
        noisy = random_noise(img, var=intensity)
        
#This is impulsive or aleatory noise. It affects certain pixels depending on tissue characteristics and techniques applies. In this case, intensity represents the amount of noise added.
    if noise == "s&p":
        noisy = random_noise(img, mode='s&p',amount=intensity)
    
#If plot=True we plot both images, the original one and the noisy image we have created.
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.title('Original image')
        plt.subplot(122)    
        plt.imshow(noisy, cmap = "gray")
        plt.axis('off')
        plt.title('Noisy image')
        plt.show()
   
    return noisy



##NL Means filtering.

# Basic 2D non-local means filtering algorithm. We create a function in which we implement this algorithm into a noisy image. 
def nlm(noisy, h=1):
    
# We are going to compare patches of the same image. The problem is that when we place the comparing patch centered at one edged pixel, a part of the comparing patch is going to be placed outside the image. That is the reason why we need to add padding, in order to create an extra row and colums in each side.

# We have used reflected mode that means that last row value or last column value is reflected to create the padding. This produces a padding in which values are similar to the ones of the image contour. We have selected this type of padding because when we tried to add a constant one, when filtering the image, it appeared a surrounding white/black frame as our images are of very low resolution.

    pad_img = np.pad(noisy,(1, 1), mode='reflect')

    rows= len(noisy)
    columns= len(noisy[0])
    nlm_img = np.zeros([(rows),(columns)]) #This will be the matrix which will correspond to the filtered image
    
#In order to apply this filter what we need to do is to iterate all over the (3x3) patches of the image getting every time a matrix of weighted values (which will depend on the similarity between the pixel to filter v(i) and the pixel v(j)that we are comparing) that will be multiplied to the original image, then get the summation of all its the elements to finally get all the corresponding filtered pixels. The procedure is the following:
    
    for i in range(rows):
        for j in range(columns):
            e_patch = pad_img[i:i+3, j:j+3] #This will be the patch (e_patch) in which the pixel that we want to filter is centered
            W = np.zeros([(rows),(columns)]) #We will obtain a new matrix of normalized weights every time we pass to the following pixel we want to filter
            for y in range(rows):
                for z in range(columns):
                    o_patch = pad_img[y:y+3, z:z+3] #Here we extract all the patches from the image in order to compare them with e_patch
                    eu_d = np.sqrt(np.sum((e_patch - o_patch)**2)) #We calculate a unique value of the euclidian distance for every o_patch
                    w = np.exp(-eu_d/(h**2)) #Then, we calculate the weight between every o_patch and e_patch
                    W[y][z] = w #We add the values of every weight to create the corresponding matrix that will be multiplied to the image
            nor_W = W/np.sum(W) #We normalize the weigth matrix
            nlm_img[i][j] = np.sum(nor_W*noisy) #This corresponds to the position of the filtered pixel 
            
    return nlm_img

## Enhanced Non-Local Means algorithm compared to pixel itself.

#As it would be explained in the notebook, when comparing patches there will be a moment when they will correspond to the same pixel so to avoid this redundandy, we will substitute this value (which is 1 due to the zero difference of the euclidean distance) for the maximum one of the weight matrix.


def nlm_itself(noisy, h=1):
    
    pad_img = np.pad(noisy,(1, 1), mode='reflect')
    
    rows= len(noisy)
    columns= len(noisy[0])
    nlm1_img = np.zeros([(rows),(columns)])

    for i in range(rows):
        for j in range(columns):
            e_patch = pad_img[i:i+3, j:j+3]
            W = np.zeros([(rows),(columns)])
            for y in range(rows):
                for z in range(columns):
                    #Until here the procedure is the same but now we will implement this condition in order to give a zero value  to the weight when patches coincide.
                    if i == y and j == z:
                        W[y][z] = 0
                    else:
                        o_patch = pad_img[y:y+3, z:z+3]
                        eu_d = np.sqrt(np.sum(np.power(e_patch - o_patch,2))) 
                        w = np.exp(-eu_d/(h**2))
                        W[y][z] = w
            max_w = np.amax(W) #Here we select the maximum value of the weight matrix.
            #Thanks to the following iteration we are able to change the value 0 (which corresponds to zero difference between pixels) for the maximum value of the weight matrix
            for ii in range(rows):
                for jj in range(columns):
                    if W[ii][jj] == 0:
                        W[ii][jj] = max_w
            nor_W = W/np.sum(W)
            nlm1_img[i][j] = np.sum(nor_W*noisy)
            
    return nlm1_img

## Enhanced Non-Local Means algorithm - CPP.

#As it would also be explained in the notebook, with this new algorithm we will weight the original weight w(i,j) of the basic NLM filter. 

#The procedure will be the same except for the new variable mu.

def nlm_cpp(noisy, h=1, D0=1, alpha=1):
    
    pad_img = np.pad(noisy,(1, 1), mode='reflect')
    rows= len(noisy)
    columns= len(noisy[0])

    nlm_cpp_img = np.zeros([(rows),(columns)])

    for i in range(rows):
        for j in range(columns):
            e_patch = pad_img[i:i+3, j:j+3]
            W = np.zeros([(rows),(columns)])
            for y in range(rows):
                for z in range(columns):
                    if i == y and j == z:
                        W[y][z] = 0
                    else:
                        o_patch = pad_img[y:y+3, z:z+3]
                        eu_d = np.sqrt(np.sum(np.power(e_patch - o_patch,2))) 
                        w = np.exp(-eu_d/(h**2))
                        denom = 1 + np.power(np.abs(noisy[i][j]-noisy[y][z])/D0, 2*alpha)
                        mu = 1/denom #With this parameter we will weight the original weight w(i,j)
                        W[y][z] = w*mu #The weight values of the matrix will correpond to the multiplication of this new weight with the original weight.
            max_w = np.amax(W)
            for ii in range(rows):
                for jj in range(columns):
                    if W[ii][jj] == 0:
                        W[ii][jj] = max_w            
            nor_W = W/np.sum(W)
            nlm_cpp_img[i][j] = np.sum(nor_W*noisy)
            
    return nlm_cpp_img


#Anisotropic filtering 

# We create a function to apply anisotropic filtering to a noisy image with a certain parameter for iterations. Bigger the parameter of iterations, more filtered the resulting image is.
def anisotropic(noisy, niter):
    
    for ii in range(niter): # We include all the algorithm inside a for loop that is going to be used depending on the number of iterations.
        
        img_eq = exposure.equalize_hist(noisy) # Due to the bad resolution of our images, we decide to perform equalization of the histogram before applying sobel filter. Thanks to equalization, we improve the constract of our image. 
        image_noise_sobel = sobel(img_eq) # Next step, is to apply a derivative filter, in our case sobel. Thanks to previous equalization sobel is going to have a better result.
        sobel_norm = image_noise_sobel/np.max(image_noise_sobel) # We normalize the sobel image in order to enhance the contrast and also to be able to create a better histogram in which we could choose the proper threshold.

# Padding is perform in our images, because we are going to divide our image into patches and then apply smooth filtering to the image. When appling a filter/kerner to images, we need to add an extra column and row in order to fill the empty pixels of the patches of pixels at edges.
#We choose constant mode because our edges pixels are background (0 value).
        pad_sobel = np.pad(sobel_norm, (1,1), mode='constant', constant_values=0)
        pad_noise = np.pad(noisy, (1,1), mode='constant', constant_values=0)

# We go through the image rows and columns in order to extract 3x3 patches, once we have these patches, we sum the gray values of each pixeal in the patch. We create a matrix in which we add this sum of gray value. The result is a matrix with the gray values of each patch.
        rows= len(pad_sobel)
        columns= len(pad_sobel[0])
        sum_sobel = np.zeros([(rows),(columns)])
        for i in range(len(sobel_norm)):
            for j in range(len(sobel_norm[0])):
                patches = pad_sobel[i:i+3, j:j+3]
                sum_0 = sum(sum(patches))
                sum_sobel[i,j] = sum_0

#Once we have our matrix with patches gray values, we go through each column and row in order to detect patches whose gray values are higher or lower to a estimated threshold (in our case 1.8). If gray value is lower than threshold, we apply mean filter to the same patch of the noisy image. So, we filter certain pixels of noisy image. However, if gray value of sum matrix is greater than threshold, the noisy image patch that we are evaluating is going to stay equal.
#We create a matrix that is going to be our final filtered image, in which we add the results of the process explained before.

        rows1= len(sum_sobel)
        columns1= len(sum_sobel[0])
        image_filtered = np.zeros([(rows1),(columns1)])
        for m in range(len(sum_sobel)):
            for n in range(len(sum_sobel[0])):

                if sum_sobel[m, n] <= 1.8:

                    parches1 = pad_noise[m:m+3, n:n+3]
                    image_filtered[m,n] = sum(sum(parches1))/9

                else:
                    image_filtered[m,n] = pad_noise[m, n]
                    
# Finally, we establish that final image filtered is the same as the noisy image. This is done to perform the iterations, if iteration is biggere than 1, final filtered image is going to repeat the same process again.  
        noisy = image_filtered

# The return of the function will be the final filtered image.
    return image_filtered


# Implementation of Anisotropic filtering (Aula Virtual)
# Thanks to this implementation we are able to compare our filtered image with the image resulting from this implementation. It they look similar our algorithm is going to be well-done.

def anisodiff(noisy,niter=1,kappa=50,gamma=0.25,step=(1.,1.),option=1,plot_flag=True):
    
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
        img = noisy.astype('float64')
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
            plt.imshow(noisy, cmap=plt.cm.gray)
            plt.title('Noisy image'), plt.axis('off')
            plt.subplot(122)
            plt.imshow(imgout, cmap=plt.cm.gray)
            plt.title('Filtered image (Anisotropic Diffusion)'), plt.axis('off')
 
        return imgout


# Mean filter.
# It is going to be used to compare the implementation of a anisotropic filter and a smoothing filter.
# Smoothing filter reduces noise but it loses details of the images (such as edges, contract...). So, theoretically our algorithm is going to be able to reduce (less) noise but it a better way (without almost reduncing quality).
# We create a function in which we are going to apply mean filter to the noisy image. Besides we create the parameter plot, thanks to it we are going to decide to plot or not the image.

def meanfilter(noisy, plot=False):
    
    #Initializing the filter of size 3x3
    size_filter = 3 
    #The filter is divided by size_filter for normalization.
    mean_filter = np.ones((size_filter,size_filter))/np.power(size_filter,2)

    #Performing the convolution between mean_filter and our noisy image.
    img_filtered = filters.convolve(noisy, mean_filter, mode = 'reflect')

    # Display the results comparing with the noisy image if required.
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(noisy, cmap=plt.cm.gray)
        plt.title('Noisy image'), plt.axis('off')
        plt.subplot(122)
        plt.imshow(img_filtered, cmap=plt.cm.gray)
        plt.title('Filtered image(Mean filter))'), plt.axis('off')
    
    #Result of the function is the image filtered.
    return img_filtered