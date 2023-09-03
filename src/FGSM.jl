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
    
    function visualise_FGSM(adv_x, noise)
        reshaped_adv_x = reshape(adv_x, 3, 224, 224)
        a = colorview(RGB, reshaped_adv_x)
        n_reshaped = reshape(noise, 3, 224, 224)
        n = colorview(RGB, n_reshaped)
        save("FGSM_attack.jpg", a)
        save("FGSM_noise.jpg", n)
    end

    function FGSM_attack(img, epsilon_range)
        preprocessed_image = FGSM_preprocess(img);
        lower_bound, upper_bound = epsilon_range;
        preprocessed_model = model_mod.preprocess_image(img);

        true_label = model_mod.predict(preprocessed_model);
        epsilon = (lower_bound + upper_bound) / 2.0;  # Initialize epsilon before the loop
        
        while abs(upper_bound - lower_bound) > 1e-5;
            adv_x, _ = FGSM(custom_loss, preprocessed_image, true_label[2], epsilon);
            adv_x = reshape(adv_x, 224, 224, 3);
            adv_label = model_mod.predict(adv_x);
            if adv_label != true_label
                upper_bound = epsilon
            else
                lower_bound = epsilon
            end
            
            epsilon = (lower_bound + upper_bound) / 2.0  # Update epsilon within the loop
        end
        
        final_adv_x, noise = FGSM(custom_loss, preprocessed_image, true_label[2], epsilon);
        f_adv_x = reshape(final_adv_x, 224, 224, 3);
        final_adv_label = model_mod.predict(f_adv_x);
       
       
        if final_adv_label != true_label
            visualise_FGSM(f_adv_x, noise)
        else
            println("It's impossible to find an epsilon value that leads to misclassification.")
        end
        return f_adv_x
    end
    
    
    
    
    
end