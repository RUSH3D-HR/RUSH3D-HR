import numpy as np
from scipy.signal import correlate

def get_corr(im, y_offset=0, x_offset=0):
    shift_im = np.roll(im, shift=(y_offset, x_offset), axis=(0, 1))
    autocorr_value = np.dot(im.flatten(), shift_im.flatten())
    return autocorr_value

def single_snr(im, saturation=250):
    im = im.cpu().numpy()
    im = im.astype(np.float32)
    im[im < 0] = 0
    im += saturation
    cor_01 = get_corr(im, 0, 1)
    cor_02 = get_corr(im, 0, 2)

    cor_00_nf = (cor_01 + cor_02) / 2
    cor_00 = get_corr(im, 0, 0)
    # print(cor_00_nf, cor_00)
    # print("mean^2 = " + str(np.mean(im) ** 2))
    snr = (cor_00_nf - np.mean(im) ** 2) / (cor_00 - cor_00_nf)
    # print("Ratio : " + str(snr))
    # print(10 * np.log10(snr))
    return 10 * np.log10(snr)
