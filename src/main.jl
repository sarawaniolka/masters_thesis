include("model.jl")
include("FGSM.jl")
include("one_pixel.jl")
include("CW.jl")

# Load required packages
using Flux
using Images
using Flux: onecold
using .model_mod  # Import the module using its name
using .FGSM_mod
using .onepix_mod
using .CW_mod

# Load the image
image_path = "masters_thesis/original_photo2.jpg"
original_image = Images.load(image_path);
original_img = model_mod.resize_image(original_image);
preprocessed_img = model_mod.normalize_tensor!(model_mod.preprocess_image(original_img));
original_prediction = model_mod.predict(preprocessed_img)

FGSM_mod.FGSM_attack(original_img, original_prediction[2], 0.19);
FGSM_image = Images.load("masters_thesis/FGSM_attack.jpg")
preprocessed_FGSM = model_mod.normalize_tensor!(model_mod.preprocess_image(FGSM_image));
FGSM_prediction = model_mod.predict(preprocessed_FGSM)


# Randomized Initial Pixel: (151, 165, 3, 1)
#Prediction: 
#("Egyptian cat", 286)

# One pixel attack
a = onepix_mod.one_pixel_attack(original_img);
one_pixel_prediction = model_mod.predict(a)


# CW attack
CW_image = CW_mod.cw_attack(original_img, original_prediction[2])
model_mod.predict(CW_image)
