include("model.jl")
include("FGSM.jl")


# Load required packages
using Flux
using Images
using Plots
using Flux: onecold
using .model_mod  # Import the module using its name
using .FGSM_mod
using .onepix_mod

# Load the image
image_path = "original_photo.jpg"
original_image = Images.load(image_path)
original_img = model_mod.resize_image(original_image)
original_prediction = model_mod.predict(original_img)

FGSM_mod.FGSM_attack(original_img, original_prediction[2], 0.21);
FGSM_image = Images.load("FGSM_attack.jpg")
FGSM_prediction = model_mod.predict(FGSM_image)

# Randomized Initial Pixel: (151, 165, 3, 1)
#Prediction: 
#("Egyptian cat", 286)
onepix_mod.one_pixel_attack(original_img);
onepix_image = Images.load("one_pixel_attack.jpg");
one_pixel_prediction = model_mod.predict(onepix_image)