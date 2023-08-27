module FGSM_mod
    using Images
    using Colors
    using Flux
    using Flux: params
    include("model.jl")
    
    import .model_mod

    export FGSM, FGSM_preprocess, visualise_FGSM, FGSM_attack;

    model = model_mod.get_model();

    # Define the loss function (e.g., cross-entropy)
    function custom_loss(x, y)
        return Flux.crossentropy(Flux.softmax(model(x)), y)
    end


    # Define the FGSM function
    function FGSM(loss, x, y, ϵ)
        grads = gradient(() -> loss(x, y), params([x]))
        peturbation = Float32(ϵ) * sign.(grads[x])
        x_adv = clamp.(x + Float32(ϵ) * sign.(grads[x]), 0, 1)
        noise = peturbation .+ 0.5
        return x_adv, noise
    end
    
    function FGSM_preprocess(original_image)
        img_array = channelview(original_image)
        img_array .= (img_array .- minimum(img_array)) / (maximum(img_array) - minimum(img_array))
        channels, height, width = size(img_array)
        image = reshape(img_array, (width, height, channels, 1))
        return image
    end
    
    function visualise_FGSM(adv_x)
        reshaped_adv_x = reshape(adv_x, 3, 224, 224)
        a = colorview(RGB, reshaped_adv_x)
        save("FGSM_attack.jpg", a)
    end

    function FGSM_attack(img, true_label_index, ϵ)
        preprocessed_image = FGSM_preprocess(img)
        adv_x, perturbation = FGSM(custom_loss, preprocessed_image, true_label_index, ϵ)
        visualise_FGSM(adv_x)
    end
end