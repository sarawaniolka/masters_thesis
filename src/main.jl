# Include the necessary modules
include("model.jl")
include("FGSM.jl")
include("one_pixel.jl")
include("CW.jl")

# Load required Julia packages
using Flux
using Images
using .model_mod 
using .FGSM_mod
using .onepix_mod
using .CW_mod

# Load the image
image_path = "original_photo.jpg"
original_image = Images.load(image_path);

# Preprocess the image
original_img = model_mod.resize_image(original_image)
preprocessed_image = model_mod.preprocess_image(original_img);
save("original_image.jpg", original_img);  # Save the resized original image

# Original prediction
original_prediction = model_mod.predict(preprocessed_image);
println("Original prediction: ", original_prediction)

# FGSM attack
epsilon_range = (0.0, 1)  # Specify the initial range for epsilon search
FGSM_data, epsilon = FGSM_mod.FGSM_attack(original_img, epsilon_range);
FGSM_prediction = model_mod.predict(FGSM_data);
println("Prediction after FGSM attack: ", FGSM_prediction)
println("Chosen epsilon: ", epsilon)

# CW attack
adv, noise = CW_mod.CW_attack(original_img, 282, 1.0, 0.0001, 100, 1.5);
CW_prediction = model_mod.predict(adv);
println("Prediction after CW attack: ", CW_prediction);

# One-pixel attack
onepix_image, pixels = onepix_mod.one_pixel_attack(original_img, 100, 800);
one_pixel_prediction = model_mod.predict(onepix_image);
println("Prediction after pixel attack: ", one_pixel_prediction)
