import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct, fft, ifft

def gen_laplace(img):
    '''Generate the LaPlacian Pyramid Stack of an Image
        inputs: 
            img: np.array containing an image of any size, with any number of color channels
        outputs:
            flat_lp: (Nx1) array containing the 6 LaPlacian Pyramids on after another
            sizes: sizes of each image in the LaPlacian Pyramid Stack'''
    # generate Gaussian pyramid 
    G = img.copy()
    gp = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gp.append(G)

    #collect the sizes before I flatten
    sizes = []
    # generate Laplacian Pyramid
    lp = [gp[5]]
    sizes.append(gp[5].shape)
    for i in range(5,0,-1):
        GE = cv.pyrUp(gp[i], dstsize = (gp[i-1].shape[1], gp[i-1].shape[0]))
        L = cv2.subtract(gp[i-1],GE)
        lp.append(L)
        sizes.append(L.shape)
    flat_lp = []
    #flatten for sake of processing
    for el in lp:
        el = el.flatten()
        flat_lp.append(el)
    flat_lp = [ x for xs in flat_lp for x in xs]
    return flat_lp, sizes

def filter_im(input, low, high, apply_dct = True):
      '''Applies a Temporal BandPass Filter on sequence of images
        inputs: 
            input: a (30, 3, N) array conataining the output of gen_laplace for each color channel 
            and frame
            low: lower frequency bound of bandpass filter (i.e. .8)
            high: upper frequency bound of bandpass filter (i.e. 1)
            apply_dct: whether or not to apply DCT filtering on the image
        outputs:
        final: (30, 3, N) array containing the 6 LaPlacian Pyramids one after another, as the 
            output of gen_laplace'''
    
    # Get the size of the input
    dimensions = input.shape
    n = dimensions[0]
    dn = len(dimensions)

    #construct mask
    freq = np.arange(0, n) 
    freq = freq / n
    mask = (freq > low) & (freq < high)
    dim = list(dimensions)
    dim.reverse()
    dim[-1] = 1
    mask = np.tile(mask, tuple(dim)).T

    #apply DCT
    if apply_dct:
        # Apply DCT along the time axis
        f = dct(input, axis=0, norm = 'ortho')

        # make frequencies outside of band 0
        f[~mask] = 0
        
        # Perform inverse DCT to get the filtered signal
        filtered = idct(f, axis=0,norm = 'ortho')

        #extract the real parts of DCT
        final = np.real(filtered)

        return final
    return mask

def amplify_pyr(num_levels, alpha, lambda_c, exaggeration_factor, final):
    '''Applies the amplification factor to magnify the qualities of interest
        inputs: 
            numLevels: number of pyramids in the stack
            alpha: amplification factor, e.g. 100
            lambda_c: spatial cutoff parameter, the spatial value for which we wil not amplify 
            exaggeration_factor: how much we want to magnify the effect for the purposes of 
            visualization
            final: output of filter_im
        outputs:
            final: now amplified (30, 3, N) array containing the 6 LaPlacian Pyramids one after 
            another, as the output of gen_laplace'''
    
    delta = lambda_c/8/(1+alpha)
    lamda = np.sqrt(720**2 + 1080**2)/3
    ind = lp_stack[0,0,:].shape[0]
    #amplify for each pyramid in stack
    for level in range(5,-1, -1):
          indices = ind - np.prod(sizes[level])
          # compute modified alpha for this level
          alpha_old = lamda/delta/8 - 1;
          alpha_old = currAlpha*exaggeration_factor
              
          if (level == num_levels or level == 1): # ignore the highest and lowest frequency band
              final[:, :, indices+1:ind] = 0
          elif (alpha_old > alpha):  # lambda bigger than lambda_c
              final[:, :, indices+1:ind] = alpha*final[:, :, indices+1:ind]
          else:
              final[:, :, indices+1:ind] = alpha_old*final[:, :, indices+1:ind]
          
          ind = indices
          lamda = lamda/2; 
    return final

def undo_laplace(flat_lp, sizes):
      '''Undoes the laplacian stack, merging all into one image
    inputs: 
        flat_lp: a (30, 3, N) array containing the output of gen_laplace for each color channel 
        and frame
        sizes: sizes of each image in the LaPlacian Pyramid Stack, output of gen_laplace
    outputs:
    ls: output image, reshaped to its original dimensions'''
    
    # undo Laplacian Pyramid
    ls = flat_lp[0:np.prod(sizes[0])].reshape(sizes[0])
    start_ind = np.prod(sizes[0])
    for i in list(range(1,6)):
        ls = cv.pyrUp(ls, dstsize = (sizes[i][1], sizes[i][0]))
        lp = flat_lp[start_ind:start_ind+ np.prod(sizes[i])].reshape(sizes[i])
        ls = ls + lp
        start_ind = start_ind + np.prod(sizes[i])
    return ls


if __name__ == "__main__":
    #apply laplacian pyramid to sapiens processed data
    lp_sequence = []
    frames = ["frame"+str(x) for x in range(30)]
    for frame in frames:
        img_path = "/processed_data/"+frame + ".png"
        img = cv2.imread(img_path)
        colors = []
        for i in range(3):
            lp, sizes = gen_laplace(img[:,:,i])
            colors.append(lp)
        lp_sequence.append(colors)
    lp_stack = np.array(lp_sequence)

    #filter and amplify each image in lplacin pyramid stack
    final = filter_im(lp_stack, .1, 10)
    final = amplify_pyr(6, 100, 1000, 2, final)

    #undo laplacian pyramid, merging the stack into one image
    full_seq = []
    for f in range(30):
        colors = []
        for c in range(3):
            im = undo_laplace(final[f, c, :], sizes)
            colors.append(im)
        full_seq.append(colors)

    #move axes to take shape (N,M,3,2)
    ims = np.moveaxis(np.moveaxis(np.array(full_seq), 0, 3), 0, 2)

    #save frame in directory
    os.mkdir('final_data')
    for x in range(30):
        cv2.imwrite("final_data/frame" + str(x) + ".png", ims[:,:,:, x])