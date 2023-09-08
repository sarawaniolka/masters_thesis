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
        return Flux.crossentropy(Flux.softmax(model(Flux.unsqueeze(x, 4))), y)
    end


    # Define the FGSM function
    function FGSM(loss, x, y, ϵ)
        grads = gradient(() -> loss(x, y), params([x]))
        peturbation = Float32(ϵ) * sign.(grads[x])
        x_adv = x + Float32(ϵ) * sign.(grads[x])
        noise = peturbation .+ 0.5
        return x_adv, noise, ϵ
    end
    
    function visualise_FGSM(adv_x, noise)
        reshaped_adv_x = permutedims(adv_x, [3, 1, 2])

        # Normalize the pixel values to the [0, 1] range
        min_val = minimum(reshaped_adv_x)
        max_val = maximum(reshaped_adv_x)
        image_data = (reshaped_adv_x .- min_val) / (max_val - min_val)

        # Create an image from the normalized data
        img = colorview(RGB, image_data)

        n_reshaped = permutedims(noise, [3, 1, 2])
        min_val = minimum(n_reshaped)
        max_val = maximum(n_reshaped)
        image_data = (n_reshaped .- min_val) / (max_val - min_val)
        n = colorview(RGB, n_reshaped)

        save("attacks_visualised/FGSM_attack.jpg", img)
        save("attacks_visualised/FGSM_noise.jpg", n)
    end

    function FGSM_attack(img, epsilon_range)
        preprocessed_image = model_mod.preprocess_image(img);
        lower_bound, upper_bound = epsilon_range;

        true_label = model_mod.predict(preprocessed_image);
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
        
        final_adv_x, noise, epsilon = FGSM(custom_loss, preprocessed_image, true_label[2], epsilon);
        f_adv_x = reshape(final_adv_x, 224, 224, 3);
        final_adv_label = model_mod.predict(f_adv_x);
       
       
        if final_adv_label != true_label
            visualise_FGSM(final_adv_x, noise)
        else
            println("It's impossible to find an epsilon value that leads to misclassification.")
        end
        return f_adv_x, epsilon
    end
    
    
    
    
    
end