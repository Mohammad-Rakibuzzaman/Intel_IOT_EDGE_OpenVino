def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2: (fh + crop_size[1]) // 2,
           (fw - crop_size[0]) // 2: (fw + crop_size[0]) // 2, :]
