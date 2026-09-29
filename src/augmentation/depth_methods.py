import numpy as np

def raw_depth_generation(depth_model, image, process_res, process_res_method):
    '''
        This function generates a raw depth map from an input image using a pre-trained depth estimation model.
        It takes an image and returns a depth map: a relative distance representation of the scene in the image. 
        The depth map is a 2D array where each pixel value represents the estimated distance from the camera to the object in the scene.
        Higher values in the depth map correspond to objects that are further away, while lower values correspond to objects that are closer.
        Each depth estimation is independent, raw values cannot be directly compared across different images, as they are relative to the specific scene and camera parameters of each image.
        Parameters:
            - depth_model: The pre-trained depth estimation model.
            - image: The input image for which the depth map is to be generated.
            - process_res: Target size for DA3 preprocessing (e.g. 504); H and W are then brought to multiples of 14.
            - process_res_method: How DA3 reaches process_res and the multiples of 14.
                Use "upper_bound_resize" (long side = process_res, whole image kept).
                Avoid "*_crop": it cuts border pixels and breaks alignment with the mask.
        Returns:
            - depth: float32 2D numpy array at DA3 NATIVE resolution (long side ~process_res,
                sides multiple of 14), NOT at the original image size.
    '''

    # Run inference of depth
    # we pass only one image(as list) at a time
    # we dont pass multiple images because multi-view models use lists of images as different views of the same scene
    prediction = depth_model.inference(
        [image],

        # process_res and process_res_method are passed explicitly (from yaml), not left to DA3 defaults:
        # see the .yaml for the meaning of each option
        process_res=process_res, 
        process_res_method=process_res_method 
        )
    
    depth = np.asarray(prediction.depth[0]).astype(np.float32)

    if not np.isfinite(depth).all():
        raise ValueError("Depth map contains non-finite values (NaN or Inf). Please check the input image and model.")

    return depth