# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

 ## Program:
```
Name: S.Harika
Reg no:212224240155
```
```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceimage=cv2.imread("Photo.jpg")
plt.imshow(faceimage[:,:,::-1]);plt.title("face")
```
<img width="451" height="462" alt="Screenshot 2025-09-07 190945" src="https://github.com/user-attachments/assets/f956c1bc-4309-4496-a954-c2e35dc34385" />

```
faceimage.shape
```
<img width="178" height="29" alt="Screenshot 2025-09-07 191039" src="https://github.com/user-attachments/assets/da5dc228-142e-4e80-bd81-22c39d3a8a00" />

```
#resized_faceImage.shape
faceimage.shape
```

<img width="191" height="41" alt="Screenshot 2025-09-07 191115" src="https://github.com/user-attachments/assets/fdf623cd-92bb-4cf2-b115-811d37355cd1" />

```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glasspng=cv2.imread('sun.png',-1)
plt.imshow(glasspng[:,:,::-1]);plt.title("GLASSPNG")
```

<img width="582" height="285" alt="Screenshot 2025-09-07 191206" src="https://github.com/user-attachments/assets/88ce3950-b331-493d-bd50-231a3bfa3ea8" />

```
# Resize the image to fit over the eye region
glasspng=cv2.resize(glasspng,(170,80))
print("image Dimension={}".format(glasspng.shape))
```

<img width="261" height="36" alt="Screenshot 2025-09-07 191255" src="https://github.com/user-attachments/assets/1bdf7853-8926-4db4-b10c-e3cebf2e5e64" />

```
import cv2
import matplotlib.pyplot as plt

# Load sunglasses (only BGR since no alpha channel exists in your file)
glasspng = cv2.imread("sun.png")

# Split BGR channels
b, g, r = cv2.split(glasspng)
glass_bgr = cv2.merge((b, g, r))

# Convert to grayscale to prepare alpha mask
gray = cv2.cvtColor(glasspng, cv2.COLOR_BGR2GRAY)

# Threshold to create alpha mask (tune threshold=240 depending on bg color)
_, glass_alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

print("BGR shape:", glass_bgr.shape)
print("Alpha shape:", glass_alpha.shape)

# Show sunglasses BGR
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(glass_bgr, cv2.COLOR_BGR2RGB))
plt.title("Sunglass BGR")
plt.axis("off")

# Show generated alpha mask
plt.subplot(1,2,2)
plt.imshow(glass_alpha, cmap="gray")
plt.title("Generated Alpha Mask")
plt.axis("off")

plt.show()
```

<img width="537" height="174" alt="Screenshot 2025-09-07 191330" src="https://github.com/user-attachments/assets/3a6cd607-98aa-4e22-848b-83f5605da767" />

```
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
facewithglassesnaive=faceimage.copy()
# Replace the eye region with the sunglass image
facewithglassesnaive[90:150, 110:250]=glassbgr
plt.imshow(facewithglassesnaive[...,::-1])
```

<img width="419" height="438" alt="Screenshot 2025-09-07 191406" src="https://github.com/user-attachments/assets/77976e57-ed91-4f23-be42-3b8d0ad683c0" />

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load images ----------------
faceimage = cv2.imread("Photo.jpg")   # your face image
glasspng  = cv2.imread("sun.png", cv2.IMREAD_UNCHANGED)  # sunglasses (with/without alpha)

if faceimage is None:
    raise FileNotFoundError("❌ Could not load faceimage.png")
if glasspng is None:
    raise FileNotFoundError("❌ Could not load sunglass.png")

# ---------------- Process sunglasses ----------------
if glasspng.shape[2] == 4:  # has alpha channel
    b, g, r, a = cv2.split(glasspng)
    glassbgr   = cv2.merge((b, g, r))   # sunglasses only
    glassmask1 = a                      # alpha channel
else:  # no alpha channel → make mask
    b, g, r = cv2.split(glasspng)
    glassbgr = cv2.merge((b, g, r))
    gray = cv2.cvtColor(glassbgr, cv2.COLOR_BGR2GRAY)
    _, glassmask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# ---------------- Eye region coordinates ----------------
y1, y2 = 90, 150
x1, x2 = 110, 250

# Resize glasses + mask to fit ROI
glassbgr   = cv2.resize(glassbgr, (x2-x1, y2-y1))
glassmask1 = cv2.resize(glassmask1, (x2-x1, y2-y1))

# Make 3-channel mask for blending
glassmask   = cv2.merge((glassmask1, glassmask1, glassmask1))
glassmask   = glassmask.astype(float) / 255.0  # scale 0-1

# ---------------- Extract Eye ROI ----------------
eyeroi = faceimage[y1:y2, x1:x2].astype(float)

# ---------------- Masked regions ----------------
maskedeye   = (eyeroi * (1 - glassmask)).astype(np.uint8)
maskedglass = (glassbgr * glassmask).astype(np.uint8)

# ---------------- Final augmented region ----------------
eyeroifinal = cv2.add(maskedeye, maskedglass)

# Put it back into the face
face_with_glasses = faceimage.copy()
face_with_glasses[y1:y2, x1:x2] = eyeroifinal

# ---------------- Show results ----------------
plt.figure(figsize=(15,5))
plt.subplot(141); plt.imshow(eyeroi[...,::-1].astype(np.uint8)); plt.title("Original Eye ROI")
plt.subplot(142); plt.imshow(maskedeye[...,::-1]); plt.title("Masked Eye Region")
plt.subplot(143); plt.imshow(maskedglass[...,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(144); plt.imshow(face_with_glasses[...,::-1]); plt.title("Augmented Eye + Sunglass")
plt.show()
```

<img width="880" height="284" alt="Screenshot 2025-09-07 191508" src="https://github.com/user-attachments/assets/b4b43368-0e49-4e62-89c7-c5bc19c60670" />

```
# Replace the eye ROI with the output from the previous section
facewithglassesarithmetic[90:150,110:250]=eyeroifinal

# Display the final result
plt.figure(figsize=[10,10]);
plt.subplot(121);plt.imshow(faceimage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(facewithglassesarithmetic[:,:,::-1]);plt.title("With Sunglasses");
```

<img width="757" height="441" alt="Screenshot 2025-09-07 191549" src="https://github.com/user-attachments/assets/e316adcb-9e92-4486-881b-26a8bc517f01" />

### Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.
  
Feel free to fork, contribute, or customize this project for your creative needs!
