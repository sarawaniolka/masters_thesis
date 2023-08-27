include("model.jl")
include("FGSM.jl")
include("one_pixel.jl")

# Load required packages
using Flux
using Images
using Flux: onecold
using .model_mod  # Import the module using its name
using .FGSM_mod
using .onepix_mod

# Load the image
image_path = "masters_thesis/original_photo.jpg"
original_image = Images.load(image_path);
original_img = model_mod.resize_image(original_image);
preprocessed_img = model_mod.preprocess_image(original_img);
original_prediction = model_mod.predict(preprocessed_img)

FGSM_mod.FGSM_attack(original_img, original_prediction[2], 0.21);
FGSM_image = Images.load("FGSM_attack.jpg")
preprocessed_FGSM = model_mod.preprocess_image(FGSM_image);
FGSM_prediction = model_mod.predict(preprocessed_image)


# Randomized Initial Pixel: (151, 165, 3, 1)
#Prediction: 
#("Egyptian cat", 286)

# One pixel attack
a = onepix_mod.one_pixel_attack(original_img);
one_pixel_prediction = model_mod.predict(a)
