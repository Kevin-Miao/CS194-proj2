import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import *
import matplotlib.pyplot as plt
import cv2
import scipy
from functions import *
from align_image_code import align_images

def crop_edges(im, percent):
    """Cropping the edges of an images
    im: Image of which the edges need to be cropped
    percent: Percentage (in decimal notation) of how much percent of each side needs to be cropped.

    Return: Cropped Image"""
    if percent > 0.5:
        print('Error')
    x_cut = int(im.shape[1] * percent)
    y_cut = int(im.shape[0] * percent)
    return im[y_cut:-y_cut, x_cut:-x_cut]

def loader(imname, gray=False):
    """
    Image loader
    imname: (String) Filename of image
    gray: (Bool, default: False) Image is in grayscale

    Returns image
    """
    im = skio.imread(imname, as_gray=gray)

    # Convert to Double
    im = sk.img_as_float(im)
    return im


def threshold(i, gradient):
    """
    Removes gradients that are within the -i<0<i range. The non zero values of the gradient are then converted to one for easier visibility.
    i: Threshold
    gradient: The gradient matrix

    Returns discretized image with noise removal.
    """
    testje = gradient.copy()
    testje[(testje < i) & (testje > -1*i )] = 0
    testje[(testje != 0)] = 1
    skio.imshow(testje, cmap=plt.cm.gray)
    return testje


def straigthen(imname, crop_edge=0.10, rotations = np.arange(-10, 11),show=True, hist=False, hist_saving=False, hist_show =False):
    """
    Image straightening.
    imname: File name of image
    crop_Edge: How much percent of the edge needs to be cropped (decimal)
    Rotations: (array) what angles need to be tried out
    *kwargs:
    show/hist/hist_show are for testing purposes

    Returns straigthened image

    """




    imname = imname

    # Read Image
    color = skio.imread(imname)
    img = skio.imread(imname, as_gray=True)

    # Convert to Double
    img = sk.img_as_float(img)

    best_rotation, best_score = None, None

    for deg in rotations:
        print(deg, sep=' ', end='', flush=True)
        if deg == 0:
            continue
        rota = np.clip(scipy.ndimage.interpolation.rotate(img, angle=deg), a_max=1, a_min=0)
        rotated = crop_edges(rota, crop_edge)
        gauss1d = cv2.getGaussianKernel(20, 3)
        gauss2d = gauss1d @ gauss1d.T
        Dx = convolve2d(gauss2d, np.matrix([1,-1]))
        x_gradi = convolve2d(rotated, Dx, mode='same')
        Dy = convolve2d(gauss2d, np.matrix([1,-1]).T)
        y_gradi = convolve2d(rotated, Dy, mode='same')
        angles = np.matrix.flatten(np.round(np.degrees(np.arctan2(-1 * y_gradi, x_gradi))))

        if hist:
            if hist_show:
                plt.hist(angles);
            if hist_saving:
                plt.savefig(f"outputs/1.3-Straigthening-Histograms/Rotation_Angles_{deg}")
            plt.clf()
        num = (np.count_nonzero(angles == 90)+ np.count_nonzero(angles == -90)) / len(angles)
#         print(num)
#         print(deg)

        if best_score == None or num > best_score:
#             print(deg)
            best_rotation = deg
            best_score = num


    returning = scipy.ndimage.interpolation.rotate(color, angle=best_rotation)

    if show:
        skio.imshow(returning)
        print(f"Straightening accomplished by rotating with degree {best_rotation}")

    return returning

def three_d_convolve(im, kernel):
    """
    Performs 3D convolutions with one kernel.
    im: 3D pictured
    kernel: 2D kernel

    Returns 3D kernel where each one of the last dimension is convoluted with the kernel.
    """
    im_r = im[:,:,0].copy()
    im_g = im[:,:,1].copy()
    im_b = im[:,:,2].copy()
    blur_im = np.dstack([convolve2d(x, kernel, mode='same') for x in [im_r, im_g, im_b]])
    return blur_im

def sharpener(imname, alpha, savename='', show=True, grey=False, crop_edge = None, clip=True, gaus_ksize = 20, gaus_std = 3):
    """
    Returns sharpened image (i.e. where larger frequencies are amplified with a factor alpha)
    imname: Filename
    alpha: strength for high frequencies to be added back
    savename: Output filename
    grey: (Bool) Greyscale image
    crop_edge: factor that edges need to be cropped with (default= None)
    gaus_ksize: Kernel size for gaussian filter
    gaus_Std: Std for gaussian filter

    Returns sharpened image
    """



    # Read Image
    im = skio.imread(imname)

    # Convert to Double
    im = sk.img_as_float(im)

    #Kernel
    gauss1d = cv2.getGaussianKernel(gaus_ksize, gaus_std)
    gauss2d = gauss1d @ gauss1d.T

    if ~grey:
        blur_im = three_d_convolve(im, gauss2d)
    else:
        blur_im = convolve2d(im, gauss2d, mode='same')

    high_fq_im = im.copy()
    if ~grey:
        high_fq_im[:,:,0] = im[:,:,0] -blur_im[:,:,0]
        high_fq_im[:,:,1] = im[:,:,1] -blur_im[:,:,1]
        high_fq_im[:,:,2] = im[:,:,2] -blur_im[:,:,2]
    output = im + alpha * high_fq_im

    if clip:
        output = np.clip(output, a_min = 0, a_max = 1)

    if crop_edge != None:
        output = crop_edges(output, crop_edge)

    if show:
        skio.imshow(output)

    if savename == '':
        savename = f'outputs/{imname}_sharpened.jpg'
    skio.imsave(savename, output)

def gaussian_blur(im, value, size=False):
    """
    Performs gausian blurring on an image.

    im: image to be blurred (can be grey or RGB)
    value: (if size=False) -> Standard deviation for Kernel
    (if size=true) -> Kernel width
    """
    if len(im.shape) == 2:
        gauss2d = gaussian_kernel(value, size)
        return convolve2d(im, gauss2d, mode="same")
    else:
        result = im.copy()
        for i in range(0,3):
            gauss2d = gaussian_kernel(value, size)
            result[:,:,i] = convolve2d(im[:,:,i], gauss2d, mode='same')
        return result


def alignment_pic(ima1='nutmeg.jpg', ima2='DerekPicture.jpg'):
    """Align two pictures with each other. Image 1 is rotated to be aligned with image 2.
    ima1, ima2: (2D/3D Matrices)

    Return 2 2D/3D Matrices of alignment.


    Note: make sure to toggle `matplotlib.use('TkAgg') or another GUI tool to manually select alignment"""
    from align_image_code import align_images

    # high sf
    im2 = plt.imread(ima2)/255.

    # low sf
    im1 = plt.imread(ima1)/255

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)
    return im1_aligned, im2_aligned

def gaussian_kernel(value, size=False):
    """Calculate the gaussian kernel filter.
    Default is that size = False. That means that the value passed in is a standard deviation.
    If size is True, then that means we pass in a kernel width/height and sigma will be precalculated. Dependencies of this functions are the gaussian blur function.
    Value: (Int) Either standard deviation or desired kernel width/height
    Size: (Bool) True: `value` is kernel width; False: 'Value' is the gaussian sigma

    Returns: (2D Matrix) Gaussian Filter """
    if size:
        size = value
        standard_deviation = int(size/6)
        gauss1d = cv2.getGaussianKernel(value, standard_deviation)
        gauss2d = gauss1d @ gauss1d.T
        return gauss2d
    else:
        standard_deviation = value
        size = int(standard_deviation) * 6
        gauss1d = cv2.getGaussianKernel(size, standard_deviation)
        gauss2d = gauss1d @ gauss1d.T
        return gauss2d

def hybrid_images(im1, im2, sigma1, sigma2, debug=False, align=True, reverse_align=False, avg=False):
    """im1, im2: (2D/3D matrix or filename) Input images (Image 1 is high frequency, Image 2 is low frequency)
    sigma1, sigma2: (int) Standard Deviations to be applied to obtain their gaussian blurs
    debug: debugging mode
    align: (Bool) if alignment is desired
    reverse_align: (Bool) Generally, im1 is aligned to be rotated with im2. If the reverse alignment is desired, please enter True

    Returns: (2D/3D matrix) of hybrid image"""
    if reverse_align and align:
        im1, im2 = im2, im1

    if align and type(im1) != str:
        import matplotlib
        matplotlib.use('TkAgg')
    #         assert im1.shape == im2.shape, "The two matrices are not the same size"
        im1_aligned, im2_aligned = align_images(im1, im2)

    if align and type(im1) == str:
        import matplotlib
        matplotlib.use('TkAgg')
        im1_aligned, im2_aligned = alignment_pic(im1, im2)


    if reverse_align and align:
        im1_aligned, im2_aligned = im2_aligned, im1_aligned

    if len(im1_aligned.shape) == 3:
        blur = im1_aligned.copy()
        highfq = im2_aligned.copy()
        for i in range(0,3):
            gauss2d1 = gaussian_kernel(sigma1)
            gauss2d2 = gaussian_kernel(sigma2)
            blur[:,:,i] = convolve2d(im2_aligned[:,:,i], gauss2d1, mode='same')
            highfq[:,:,i] = im1_aligned[:,:,i] - convolve2d(im1_aligned[:,:,i], gauss2d2, mode='same')
        hybrid = np.clip(blur+highfq, a_min=0, a_max=1)
    else:
        gauss2d1 = gaussian_kernel(sigma1)
        gauss2d2 = gaussian_kernel(sigma2)
        blur = convolve2d(np.mean(im2_aligned, axis=2), gauss2d1, mode='same')
        highfq = np.mean(im1_aligned, axis=2) - convolve2d(np.mean(im1_aligned, axis=2), gauss2d2, mode='same')
        hybrid = np.clip(blur+highfq, a_min=0, a_max=1)
    if debug:
        skio.imshow(hybrid)
    if avg:
        hybrid = np.clip(blur+highfq/2, a_min=0, a_max=1)
    return hybrid

def hybrid_images_fourier(im1, im2, sigma1, sigma2, debug=False, align=True, reverse_align=False, avg=False):
    """im1, im2: (2D/3D matrix or filename) Input images (Image 1 is high frequency, Image 2 is low frequency)
    sigma1, sigma2: (int) Standard Deviations to be applied to obtain their gaussian blurs
    debug: debugging mode
    align: (Bool) if alignment is desired
    reverse_align: (Bool) Generally, im1 is aligned to be rotated with im2. If the reverse alignment is desired, please enter True

    Returns: (2D/3D matrix) of hybrid image"""
    fourier = []
    lowhigh =[]
    if reverse_align and align:
        im1, im2 = im2, im1

    if align and type(im1) != str:
        import matplotlib
        matplotlib.use('TkAgg')
    #         assert im1.shape == im2.shape, "The two matrices are not the same size"
        im1_aligned, im2_aligned = align_images(im1, im2)

    if align and type(im1) == str:
        import matplotlib
        matplotlib.use('TkAgg')
        im1_aligned, im2_aligned = alignment_pic(im1, im2)


    if reverse_align and align:
        im1_aligned, im2_aligned = im2_aligned, im1_aligned

    if len(im1_aligned.shape) == 3:
        blur = im1_aligned.copy()
        highfq = im2_aligned.copy()
        for i in range(0,3):
            gauss2d1 = gaussian_kernel(sigma1)
            gauss2d2 = gaussian_kernel(sigma2)
            blur[:,:,i] = convolve2d(im2_aligned[:,:,i], gauss2d1, mode='same')
            highfq[:,:,i] = im1_aligned[:,:,i] - convolve2d(im1_aligned[:,:,i], gauss2d2, mode='same')
        hybrid = np.clip(blur+highfq, a_min=0, a_max=1)
        fourier += [np.log(np.abs(np.fft.fftshift(np.fft.fft2(highfq))))]
        fourier += [np.log(np.abs(np.fft.fftshift(np.fft.fft2(blur))))]
        lowhigh += [blur]
        lowhigh += [highfq]
    else:
        gauss2d1 = gaussian_kernel(sigma1)
        gauss2d2 = gaussian_kernel(sigma2)
        blur = convolve2d(np.mean(im2_aligned, axis=2), gauss2d1, mode='same')
        highfq = np.mean(im1_aligned, axis=2) - convolve2d(np.mean(im1_aligned, axis=2), gauss2d2, mode='same')
        hybrid = np.clip(blur+highfq, a_min=0, a_max=1)
    if debug:
        skio.imshow(hybrid)
    if avg:
        hybrid = np.clip(blur+highfq/2, a_min=0, a_max=1)
    return hybrid, fourier, lowhigh


def gaussian_stack(im, level, sigma, debug=False):
    """
    Returns gaussian stack

    im: image (matrix)
    level: number of levels in stack
    sigma: standard deviation for gaussian blur

    Note: item[0] is always the original picture
    Grey/Color detection is automatic

    Return (array of elements)
    """
    result = [im]

    for i in np.arange(1, level + 1):
        result += [gaussian_blur(result[-1], sigma)]
    return np.array(result)


def laplacian_stack(im, level, sigma, debug=False):
    """    Returns laplacian stack

        im: image (matrix)
        level: number of levels in stack
        sigma: standard deviation for gaussian blur

        Note: item[0] is always the original picture
        Grey/Color detection is automatic

        Return (array of elements)
        """
    gausstack = gaussian_stack(im, level, sigma)
    result = []
    for i in range(len(gausstack)-1):
        result += [(gausstack[i] - gausstack[i+1])]
    result += [gausstack[-1]]
    return np.array(result)

def multi_resolution_blending(image1, image2, mask, sigma, level, debug=False, gray=False, clip=False):
    """
    This function performs the multiresolution blending process using a LaPlacian stack of two images and a mask that determines what is being blended.
    Make sure to have the mask representing the part that shall be represented by image 1.
    image1, image2: (Matrix) Grey/Color images that are being blended
    mask: (Matrix) Same shape as image1, image2, where mask is representing ones of what part needs to be incorporated in the hybridized picture
    sigma: (int) The standard deviation for the gaussian blur used for the laplacian stack
    level: (int) Number of levels in the laPlacian stack
    debug (Bool) Debugging Mode
    gray: (Bool) If picture being put in is color and this is toggled on, the output will be gray.

    Returns: (Matrix) Hybridizied picture
    """
    assert image1.shape == image2.shape == mask.shape, "One of the shapes are off. Make sure that everything has the same dimension"
    lp1 = laplacian_stack(image1, level, sigma)
    lp2 = laplacian_stack(image2, level, sigma)
    Gmask = gaussian_stack(mask, level, sigma)
    Gr = np.array(Gmask)
    La = np.array(lp1)
    Lb = np.array(lp2)

    if gray or len(image1.shape)==2:
        result = Gr * La + (1-Gr) * Lb
        if clip:
            result = np.clip(result, a_min=0, a_max=1)
        return sum(result)
#     for i in range(3):
    result = Gr * La + (1-Gr) * Lb
    if clip:
        result = np.clip(result, a_min=0, a_max=1)
    return sum(result)
