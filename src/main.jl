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
image_path = "original_photo.jpg";
original_image = Images.load(image_path);

# Preprocess the image
original_img = model_mod.resize_image(original_image)
preprocessed_image = model_mod.preprocess_image(original_img);
save("original_image.jpg", original_img);
# Original prediction
original_prediction = model_mod.predict(preprocessed_image);
println("Original prediction: ", original_prediction)

epsilon_range = (0.0, 1);  # Specify the initial range for epsilon search
FGSM_data, epsilon = FGSM_mod.FGSM_attack(original_img, epsilon_range);
FGSM_prediction = model_mod.predict(FGSM_data);
println("Prediciton after FGSM attack: ", FGSM_prediction)
println("Chosen epsilon: ", epsilon)

# CW attack
adv = CW_mod.cw_attack(original_img, 282, 0.5, 0.9, 50);
CW_prediction = model_mod.predict(adv);
println("Prediciton after CW attack: ", CW_prediction)

# Randomized Initial Pixel: (151, 165, 3, 1)
#Prediction: 
#("Egyptian cat", 286)

onepix_image, pixels= onepix_mod.one_pixel_attack(original_img, 100, 1000);
one_pixel_prediction = model_mod.predict(onepix_image)
println(pixels)
println("Prediciton after pixel attack: ", one_pixel_prediction)