import torch
import torch.nn as nn

def split_images(X, window_size):
    """
    Split X into non-overlapped patches by sliding a window
    Input:
        X           : 4-D Tensor Batch x Channel x Height x Width
        window_size : (h, w) size of sliding window
    Output:
        X_splitted  : 3-D Tensor (Batch*NB_patches, h*w, Channel), Channel as final dimension to feed QKV easily
    """
    h, w = window_size
    B, C, _, _ = X.size()
    # split image into non-overlaped patches
    X_splitted = nn.Unfold(kernel_size=window_size, stride=window_size)(X) # (Batch, C*h*w, NB_patches), (h,w): window_size
    X_splitted = X_splitted.reshape(B, C, h*w, X_splitted.size(-1)) # (Batch, C, h*w, NB_patches)
    X_splitted = X_splitted.permute(0, 3, 2, 1) # (Batch, NB_patches, h*w, C)
    X_splitted = X_splitted.reshape(-1, h*w, C) # (Batch*NB_patches, h*w, C), h*w: nb pixels in a patch
    return X_splitted

def merge_splitted_images(X_splitted, window_size, original_size):
    """
    Merge splitted images to form the orignial image
    Input:
        X_splitted      : 3-D Tensor (Batch*NB_patches, h*w, Channel)
        window_size     : (h, w) size of sliding window
        original_size   : (H, W) size of images before splitting
    Output:
        X               : 4-D Tensor Batch x Height x Width x Channel, Channel as final dimension for MLP
    """
    H, W = original_size
    h, w = window_size
    B_nbPatches, h_w, C = X_splitted.size()
    # find number of patches
    nbPatches = (H*W)//(h*w)
    B = B_nbPatches//nbPatches
    X_merged = X_splitted.view(B, nbPatches, h_w, C) # (Batch, nbPatches, h*w, C)
    X_merged = X_merged.view(B, nbPatches, h, w, C) # (Batch, nbPatches, h, w, C)
    X_merged = X_merged.view(B, H//h, W//w, h, w, C)
    X_merged = X_merged.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return X_merged

def rgb_to_ycbcr(image):
    """Convert an RGB image to YCbCr.

    Args:
        image       : 4-D tensor RGB Image (B,C,H,W) to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = .5
    y  = .299 * r + .587 * g + .114 * b
    cb = (b - y) * .564 + delta
    cr = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

def get_y_channel(output):
    y_pred, y = output
    y_pred, y = rgb_to_ycbcr(y_pred), rgb_to_ycbcr(y)
    # y_pred and y are (B, 3, H, W) and YCbCr or YUV images
    # let's select y channel
    return y_pred[:, 0, ...], y[:, 0, ...]


if __name__ == '__main__':
    B = 2
    C = 180
    H = 64
    W = 64
    h = 8
    w = 32
    nb_patches = (H*W)//(h*w)
    print("Tests")
    print("Split image")
    X = torch.rand((B, C, H, W))
    X_splitted = split_images(X, (h, w))
    assert X_splitted.size() == (B*nb_patches, h*w, C), "Error split_images"
    print("Split image: done")
    print("Merge images")
    X_merged = merge_splitted_images(X_splitted, (h, w), (H, W))
    assert (X!=X_merged).sum().item() == 0, "Error merge_splitted_images"
    print("Merge image: done")
    print("rgb_to_ycbcr")
    X = torch.rand((B, 3, H, W))
    assert rgb_to_ycbcr(X).size() == (B, 3, H, W), "Error rgb_to_ycbcr"
    print("rgb_to_ycbcr: done")







