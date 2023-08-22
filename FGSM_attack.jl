using Metalhead
using DataAugmentation
using Flux
using Flux: onecold, params, gradient, Ref
using Images
using Colors

# Stats derived from the statistics of the ImageNet dataset
DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)

# Load the image
img = Images.load("C:/Users/Sara/Desktop/photo_c.jpg")

# Apply CenterCrop augmentation
augmentations = CenterCrop((224, 224))
cropped_img = apply(augmentations, Image(img)) |> itemdata

# Convert the cropped image to tensor and normalize
normalized_data = apply(ImageToTensor() |> Normalize(DATA_MEAN, DATA_STD), Image(img)) |> itemdata

# ImageNet labels
labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))
model = ResNet(34; pretrain = true)

# Get the model's target prediction
target = onecold(model(Flux.unsqueeze(normalized_data, 4)), labels)

# Find the index associated with the target label
true_label = findfirst(label -> label == target[1], labels)

# Define the loss function (e.g., cross-entropy)
function loss(x, y)
    return Flux.crossentropy(Flux.softmax(model(x)), y)
end

ps = Flux.params(model);

# Define the FGSM function
function FGSM(model, loss, x, y; ϵ = 0.003)
    grads = gradient(() -> loss(x, y), params([x]))
    peturbation = Float32(ϵ) * sign.(grads[x])
    x_adv = clamp.(x + Float32(ϵ) * sign.(grads[x]), 0, 1)
    noise = peturbation .+ 0.5
    return x_adv, noise
end

# Preprocess the image (e.g., resize, normalize, convert to tensor)
img_array = channelview(cropped_img)
img_array .= (img_array .- minimum(img_array)) / (maximum(img_array) - minimum(img_array))
channels, height, width = size(img_array)
image = reshape(img_array, (width, height, channels, 1))
image

# Generate adversarial example using FGSM
adv_x, peturbation = FGSM(model, loss, image, true_label)

# Check the model's prediction on the original and adversarial examples
println("Original Prediction: ", onecold(model(Flux.unsqueeze(normalized_data, 4)), labels))
println("Adversarial Prediction ", onecold(model(adv_x), labels))

adv_x

# Assuming adv_x is your array of size 497×429×3×1
reshaped_adv_x = reshape(adv_x, 3, 224, 224)
colorview(RGB, reshaped_adv_x)