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

    function FGSM_attack(img, epsilon_range, max_iterations = 10, batch_size = 5)
        preprocessed_model = model_mod.preprocess_image(img)
        normalized_model = model_mod.normalize_tensor(preprocessed_model)
        true_label = model_mod.predict(normalized_model)
        
        preprocessed_image = FGSM_preprocess(img)
        
        lower_bound, upper_bound = epsilon_range
        step_size = (upper_bound - lower_bound) / max_iterations
        
        adv_x, _ = FGSM(custom_loss, preprocessed_image, true_label[2], lower_bound)
        adv_x = model_mod.normalize_tensor(reshape(adv_x, 224, 224, 3))
        
        for iteration in 1:max_iterations
            epsilon = lower_bound + step_size * iteration
            
            batch_adv_x = similar(adv_x, batch_size)
            for i in 1:batch_size
                batch_adv_x[i], _ = FGSM(custom_loss, preprocessed_image, true_label[2], epsilon)
            end
            
            batch_adv_x = model_mod.normalize_tensor(reshape(batch_adv_x, 224, 224, 3, batch_size))
            
            for i in 1:batch_size
                adv_label = model_mod.predict(batch_adv_x[:, :, :, :, i])
                
                if adv_label != true_label
                    adv_x = batch_adv_x[:, :, :, :, i]
                    break
                end
            end
            
            if adv_label != true_label
                break
            end
        end
        
        final_adv_x = model_mod.normalize_tensor(reshape(adv_x, 224, 224, 3))
        final_adv_label = model_mod.predict(final_adv_x)
        
        if final_adv_label != true_label
            visualise_FGSM(final_adv_x)
        else
            println("It's impossible to find an epsilon value that leads to misclassification.")
        end
        
        return final_adv_x
    end
      
    
    
    
end