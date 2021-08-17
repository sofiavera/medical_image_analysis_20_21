import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 


##Seed growing.

#We need a function which repeats the same procedure along all the pixels of the image that meet the condition of being inside a certain gray region. This function has the following parameters:

    # x and y - they represent the position of the central pixel
    # matrix - the image
    # boolean - this matrix will help to know which pixels have been analysed in order to do not pass by them again, it avoids an infinite loop
    # sigma_up and sigma_down - gray level range
    # mask - binary mask with the evolution of the segmentation

def expansion(x,y,matrix,boolean,sigma_up,sigma_down,mask):
    
    #the varible neighbors represents the position of the eight adjacents of the central pixel
    neighbors = [(-1,-1),(-1, 0),(-1,+1),
                (0, -1), (0, 1),
                (+1,-1),(1, 0),(+1,+1)]
    
    #loop used to iterate over the eight adjacents
    for i in range(8):
        x_new = x + neighbors[i][0] #new row position
        y_new = y + neighbors[i][1] #new column position
        
        #condition used to check if the position we are analysing is inside the image, used for adjacent pixels whose central pixel is located in the border
        if x_new >=0 and x_new <=(len(matrix)-1) and y_new >=0 and y_new <=(len(matrix[0])-1):
            
            #condition used to check if the position of the pixel has been iterated previously
            if boolean[x_new][y_new]:
                boolean[x_new][y_new] = 0 #we change the value to zero to set a False value so in future iterations, the pixel that has already been checked cannot enter inside this condition
                
                #then if not, we finally check if the pixel is inside the gray range, we segment the pixel with a white value
                if (sigma_down<= matrix[x_new][y_new] <= sigma_up) :
                    mask[x_new][y_new] = 1
                    expansion(x_new,y_new,matrix,boolean,sigma_up,sigma_down,mask) #we apply recusion as we need to repeat this procedure several times
                    
    return mask #this output mask can be referred as the incomplete mask that is being modified along the recursions or to the final binary image which represents the asked segmentations

#This is the principal function which will use the previoius funtion. It has the following parameters:
    #img - the image we want to segment
    #num_seeds - number of parts we want to segment
    #t - a value that will modified the gray range

def RegionGrowingP2(img,num_seeds,t):
    
    #line of code necessary for the use of the ginput function
    matplotlib.use('TkAgg')
    
    #selection of the initial seed
    plt.imshow(img, cmap ="gray")
    coord = plt.ginput(num_seeds) 
    
    #the matrix in which the segmentation will be implemented
    boolean_mask = np.zeros((len(img),len(img[0])))
    
    #we iterate over the number of seeds
    for i in range(len(coord)):
        
        #column and row refer to the position of the selected seeds
        column = int(coord[i][0].round())
        row = int(coord[i][1].round())
        
        print("The position of the seed number",i,"is",row,column)
        
        #we define the gray range using the value t
        seed = img[row][column]
        threshold_up = seed + t
        threshold_down = seed - t
        
        #this matrix is the one that has been explained before, the one that will be used to check which pixels has been iterated
        boolean_matrix = np.ones((len(img),len(img[0])))
        boolean_matrix[row][column] = 0
        
        boolean_mask[row][column] = 1
        
        #we iterate over the necessary pixels
        final = expansion(row,column,img,boolean_matrix,threshold_up,threshold_down,boolean_mask)
    
    return final

#this final variable correspond to the binary mask which segmented area is in white color