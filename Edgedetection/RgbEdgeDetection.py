from PIL import Image
import numpy as np
import time
# ---------------------------------------------------------------------
# Color edge detection for fun.
# Use ".convert('L')" on image for proper black white image detection.
#
# fastEdgeDetection has implemented an optimization i found smart.
# black and white only
# ---------------------------------------------------------------------
def main():
    img = (Image.open("corgi.jpg"))#.convert("L")
    img.show()

    start = time.time()
    processed_img = Image.fromarray(fastEdgeDetection(np.array(img)))
    print(f"alg took: {time.time()-start}")

    processed_img.show()

    save(processed_img)


# ---------------------------------------------------------------------
# My edge detection algorithm, default is with colors
# Could generalize but don't see a reason to
# ---------------------------------------------------------------------
def rgbEdgeDetection(img):
    ans = np.copy(img)
    brightness = 2          # typical values are 1 or 2
    fltr_ver = np.array([0.25, 0, -0.25, 0.5, 0, -0.5, 0.25, 0, -0.25]) * brightness # vertical edges
    fltr_hor = np.array([-0.25, -0.5, -0.25, 0, 0, 0, 0.25, 0.5, 0.25]) * brightness # horizontal edges
    col = img[0][0].size
    row = len(fltr_ver)

    for i in range(1, len(img) - 1):
        for j in range(1, len(img[0]) - 1):
            curr = img[i - 1:i + 2, j - 1:j + 2]
            curr_shaped = curr.reshape(row,col).T
            ans[i][j] = ((np.dot(curr_shaped, fltr_hor))**2
                        + (np.dot(curr_shaped, fltr_ver))**2)**0.5

    return ans.astype(np.uint8)


# ---------------------------------------------------------------------
# Optimized black and white edge detection. ca 0.095s runtime, about 77 times faster.
# Got the optimization from: https://craftofcoding.wordpress.com/2013/12/18/image-processing-in-python-code-efficiency/
# ---------------------------------------------------------------------
def fastEdgeDetection(img):
    fltr_lod = np.array([0.25, 0, -0.25, 0.5, 0, -0.5, 0.25, 0, -0.25])
    fltr_hor = np.array([-0.25, -0.5, -0.25, 0, 0, 0, 0.25, 0.5, 0.25])

    r = 1
    i = 0
    rN = (r * 2 + 1) ** 2.0
    nr, nc = img.shape
    im1 = np.zeros((nr - 2 * r, nc - 2 * r), dtype=np.float)
    im2 = np.zeros((nr - 2 * r, nc - 2 * r), dtype=np.float)
    for k in range(-r, r + 1):
        for l in range(-r, r + 1):
            curr = (img[r + k:nr - r + k, r + l:nc - r + l])
            im1 += curr * fltr_lod[i]
            im2 += curr * fltr_hor[i]
            i+=1
    imS = (im1**2+im2**2)**0.5
    return imS.astype(np.uint8)


# lazy save function. It works.
def save(img):
    like2save = input("would you like to save? input 1 \n")
    if like2save == "1":
        with open('img_count', 'r') as count:
            i = count.readline()
            print(f"this is i: {i}")
            if (len(i)> 0):
                i = int(i)
                img.save(f"C:/Users/kobbe/OneDrive/Skrivbord/Image manipulation/img{i}.png")
                i += 1
            else:
                i = 1
                img.save(f"C:/Users/kobbe/OneDrive/Skrivbord/Image manipulation/img{i}.png")
                i += 1
        with open('img_count', 'w') as count:
            count.write(str(i))


if __name__ == '__main__':
    main()
