import cv2

def show_image_pair(optical, thermal, dt):
    '''
    Displaying the image pair for dt ms.
    If dt is 0 then this function is blocking until a key is pressed.
    
        Parameters
    ----------
    optical : numpy.array
        Optical image
    thermal : numpy.array
        Thermal image
    dt : int
        Time for which the images are shown and the execution is blocked [ms]
    '''

    cv2.imshow("optical", optical)
    cv2.imshow("thermal", thermal)

    cv2.waitKey(dt)

def show_image_gifs(images_1, images_2, dt = 250):
    '''
    Displaying the image pairs as GIF by constantly flipping between them.
    The GIFs are displayed until the "n" key is pressed.

        Parameters
    ----------
    images_1 : list
        List of the respective first image of the GIF
    images_2 : numpy.array
        List of the respective second image of the GIF
    dt : int
        Timespan between changing the displayed image [ms]
    '''

    images = images_1
    is_1= True

    while True:
        for i, im in enumerate(images):
            cv2.imshow(str(i), im)

        k = cv2.waitKey(dt)
        if k > -1:
            k = chr(k)
            if k == 'n':
                break
        if is_1:
            is_1 = False
            images = images_2
        else:
            is_1 = True
            images = images_1
