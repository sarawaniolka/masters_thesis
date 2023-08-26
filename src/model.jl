# model.jl

module model_mod

    using Flux
    using Flux: onecold, params, gradient, Ref
    using DataAugmentation
    using Metalhead

    export DATA_MEAN, DATA_STD, load_image, preprocess_image, get_labels, get_model, get_true_label, loss, model_parameters

    # Stats derived from the statistics of the ImageNet dataset
    const DATA_MEAN = (0.485, 0.456, 0.406)
    const DATA_STD = (0.229, 0.224, 0.225)

    # Load the image
    function load_image(image_path)
        return Flux.Data.ImageDataset(image_path)
    end

    # Preprocess the image: CenterCrop, convert to tensor, and normalize
    function preprocess_image(image, crop_size=(224, 224))
        #augmentations = CenterCrop(crop_size)
        #cropped_img = apply(augmentations, Image(image)) |> itemdata
        normalized_data =  apply(ImageToTensor() |> Normalize(DATA_MEAN, DATA_STD), Image(image)) |> itemdata
        return normalized_data
    end
        
    # Load ImageNet labels
    function get_labels()
        labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))
        return labels
    end

    # Load the model
    function get_model()
        return ResNet(34; pretrain = true)
    end

    # Find the true label index
    function get_true_label(labels, target)
        true_label = findfirst(label -> label == target[1], labels)
        return true_label
    end

    # Get model parameters
    function model_parameters(model)
        return Flux.params(model)
    end

end # module
