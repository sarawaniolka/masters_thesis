include("model.jl")
include("FGSM.jl")
include("one_pixel.jl")

# Load required packages
using Flux
using Images
using Plots
using Flux: onecold
using .model_mod  # Import the module using its name
using .FGSM_mod
using .onepix_mod

# Load the image
image_path = "original_photo.jpg";
original_image = Images.load(image_path);

# Preprocess the image
original_img = model_mod.resize_image(original_image)
preprocessed_image = model_mod.preprocess_image(original_img);

# Original prediction
original_prediction = model_mod.predict(preprocessed_image);
println("Original prediction: ", original_prediction)

# FGSM attack with binary search used for optimizing epsilon
epsilon_range = (0.0, 0.4);  # Specify the initial range for epsilon search
FGSM_data = FGSM_mod.FGSM_attack(original_img, epsilon_range);
FGSM_prediction = model_mod.predict(FGSM_data);
println("Prediciton after FGSM attack: ", FGSM_prediction)

# Randomized Initial Pixel: (151, 165, 3, 1)
#Prediction: 
#("Egyptian cat", 286)
onepix_mod.one_pixel_attack(original_img);
onepix_image = Images.load("one_pixel_attack.jpg");
one_pixel_prediction = model_mod.predict(onepix_image)