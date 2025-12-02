import numpy as np
import skimage
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import os
from skimage import io, img_as_float, color
from sklearn.decomposition import PCA

#this file contains all the tested method to augment the images, for each method:
#input: tI (image to augment)
#input: mask_generator (the automatic maske generator we want to produce the masks with.
#       Can be an instance of SAM2AutomaticMaskGenerator or SamAutomaticMaskGenerator)
#output: a 3 channel augmented image

def SAMAug(tI,mask_generator):
    #generate will return the list of masks provided by SAM (and we need to iterate over them)
    masks = mask_generator.generate(tI)         
    tI=skimage.img_as_float(tI)
    #initialize the matrix                     
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    BoundaryPrior=np.zeros((tI.shape[0],tI.shape[1]))

    #masks is a list of dictionarys, one for each image segment. More information at: 
    #https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
    for mask in masks:
        thismask=mask['segmentation']
        stability_score =mask['stability_score']
        thismask_=np.zeros((thismask.shape))
        #transform thismask from a boolean matrix to a 0-1 matrix
        thismask_[np.where(thismask==True)]=1
        #give color to the masks based on the stability score
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
        #boundry_prior contains the contours of SAM's masks  
        BoundaryPrior=BoundaryPrior+find_boundaries(thismask_,mode='thick')
        BoundaryPrior[np.where(BoundaryPrior>0)]=1

    tI[:,:,1]=tI[:,:,1]+SegPrior
    tI[:,:,2]=tI[:,:,2]+BoundaryPrior

    return tI

def ourSAMAug(tI,mask_generator):
    masks = mask_generator.generate(tI)
    tI=skimage.img_as_float(tI)                     
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    BoundaryPrior=np.zeros((tI.shape[0],tI.shape[1]))
    area_prior=np.zeros((tI.shape[0],tI.shape[1]))
    for mask in masks: 
        thismask=mask['segmentation']
        stability_score =mask['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
        BoundaryPrior=BoundaryPrior+find_boundaries(thismask_,mode='thick')
        BoundaryPrior[np.where(BoundaryPrior>0)]=1

        #our modification:
        area=mask['area']/(tI.shape[0]*tI.shape[1])
        area_prior[np.where(thismask_==1)]=area_prior[np.where(thismask_==1)]+area
    tI[:,:,0]=tI[:,:,0]+area_prior
    tI[:,:,1]=tI[:,:,1]+SegPrior
    tI[:,:,2]=tI[:,:,2]+BoundaryPrior

    return tI

#remove the blue channel and replace it with the segPrior
def RG_segPrior(tI,mask_generator):
    masks = mask_generator.generate(tI)
    tI=skimage.img_as_float(tI)                     
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    for mask in masks: 
        thismask=mask['segmentation']
        stability_score =mask['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
    r = tI[:, :, 0]
    g = tI[:, :, 1]
    b = SegPrior

    modded_image= np.dstack((r,g,b))

    return modded_image

#convert the image to HSV color space, than replace the h channel with the segPrior
def SV_segPrior(tI,mask_generator):
    masks = mask_generator.generate(tI)
    tI=skimage.img_as_float(tI)                     
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    for mask in masks: 
        thismask=mask['segmentation']
        stability_score =mask['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
    
    hsv_image = color.rgb2hsv(tI)
    h = tI[:, :, 0]
    s = tI[:, :, 1]
    v = tI[:, :, 2]
    m = SegPrior
    #print(np.var(h), np.var(s), np.var(v) )

    modded_image= np.dstack((s,v,m))

    return modded_image

#remove te blue channel and replace it with the sam's masks logits 
def RG_logits(tI,mask_generator):
    #set the current image as SAM's input
    mask_generator.predictor.set_image(tI)
    #get the logits
    mask,_,logits = mask_generator.predictor.predict(return_logits=True, multimask_output=False)
    
    r = tI[:, :, 0]
    g = tI[:, :, 1]
    b = tI[:, :, 2]

    max = np.max(mask[0])
    min = np.min(mask[0])
    b = ((mask[0]-min)*255)/(max-min)
    modded_image= np.dstack((r,g,b))

    return modded_image

#reduce the dimensionality of the image using pca (3 to 2 channels) and add segPrior
def PCA_segPrior(tI,mask_generator):
    masks = mask_generator.generate(tI)
    tI=skimage.img_as_float(tI)                     
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    for mask in masks: 
        thismask=mask['segmentation']
        stability_score =mask['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
    
    
    h, w, c = tI.shape
    flat_image = tI.reshape(-1, c)

    pca = PCA(n_components=2)
    flat_image_pca = pca.fit_transform(flat_image)

    image_pca = flat_image_pca.reshape(h, w, 2)

    #scale the PCA output to [0, 255]
    image_pca_scaled = (image_pca - np.min(image_pca)) / (np.max(image_pca) - np.min(image_pca)) * 255
    image_pca_scaled = image_pca_scaled.astype(np.uint8)
    
    r =  image_pca_scaled[:, :, 0]
    g =  image_pca_scaled[:, :, 1]
    b = (255 * (SegPrior - SegPrior.min()) / (SegPrior.max() - SegPrior.min())).astype(np.uint8)

    modded_image= np.dstack((r,g,b))

    return modded_image
