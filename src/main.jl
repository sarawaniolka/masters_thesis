include("model.jl")
include("FGSM.jl")

# Load required packages
using Flux
using Images
using Plots
using Flux: onecold
using .model_mod  # Import the module using its name
using .FGSM_mod

# Load the image
image_path = "C:/Users/Sara/Desktop/photo_c.jpg"
loaded_image = Images.load(image_path)

# Preprocess the image
preprocessed_image = model_mod.preprocess_image(loaded_image);
# Load labels
labels = model_mod.get_labels();

# Load the model
model = model_mod.get_model();

# Get the target label
target_prediction = Flux.onecold(Flux.softmax(model(Flux.unsqueeze(preprocessed_image, 4))), labels);
true_label_index = model_mod.get_true_label(labels, target_prediction);

# Define loss function
function custom_loss(x, y)
    return Flux.crossentropy(Flux.softmax(model(x)), y)
end

# Calculate the loss
#current_loss = model_mod.loss(model, preprocessed_image, true_label_index)

# Preprocess the image for FGSM
preprocessed_image = FGSM_mod.FGSM_preprocess(loaded_image);
adv_x, perturbation = FGSM_mod.FGSM(custom_loss, preprocessed_image, true_label_index, 0.003);

println("Original Prediction: ", target_prediction[1])
println("Adversarial Prediction (FGSM): ", onecold(model(adv_x), labels)[1])
a = visualise_FGSM(adv_x)
save("FGSM_attack.jpg", a)