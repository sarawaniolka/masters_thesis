# model.jl

module model_mod

    using Flux
    using Images
    using Flux: onecold, params, gradient, Ref
    using DataAugmentation
    using Metalhead

    export DATA_MEAN, DATA_STD, load_image, preprocess_image, get_labels, get_model, get_true_label, loss, model_parameters, custom_loss, resize_image;

    # Stats derived from the statistics of the ImageNet dataset
    const DATA_MEAN = (0.485, 0.456, 0.406)
    const DATA_STD = (0.229, 0.224, 0.225)

    # Load the image
    function load_image(image_path)
        return Flux.Data.ImageDataset(image_path)
    end

    function resize_image(image)
        resized_image = Images.imresize(image, (224, 224), center=true)  # Resize and center the image
    end

    # Preprocess the image: convert to tensor, and normalize
    function preprocess_image(image)
        normalized_data = apply(ImageToTensor() |> Normalize(DATA_MEAN, DATA_STD), Image(image)) |> itemdata
        return normalized_data
    end
        
    # Load ImageNet labels
    function get_labels()
        labels = readlines(download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"))
        return labels
    end

    # Load the model
    function get_model()
        return ResNet(50; pretrain = true)
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

    function predict(image)
        labels = get_labels();
        model = get_model();
        target_prediction = Flux.onecold(Flux.softmax(model(Flux.unsqueeze(image, 4))), labels);
        true_label_index = get_true_label(labels, target_prediction);
        prediction = target_prediction[1];
        return prediction, true_label_index;
    end

end # module
