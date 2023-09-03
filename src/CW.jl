module CW_mod
    using Images
    using Colors
    using Flux
    using Flux: params, update!, Optimise, setup
    include("model.jl")
    include("FGSM.jl")

    import .model_mod
    import .FGSM_mod
    export CW_attack



# Define the CW loss function
    function cw_loss(delta, x, t, model, c)
        predictions = model(Flux.unsqueeze(x, 4) .+ delta) #prediction with logits
        target_logit = predictions[t]
        max_diff = -Inf
        max_i = -1

        for i in eachindex(predictions)
            if i != t
                diff = predictions[i] - target_logit
                if diff > max_diff
                    max_diff = diff
                    max_i = i
                end
            end
        end

    euclidean_dist = sqrt(sum((predictions .- target_logit).^2) + eps())
    loss = max(max_diff - euclidean_dist + c, 0.0)

    return loss
    end

    # Define the CW attack function
    function cw_attack(image, target_class, c, max_iterations)
        # Initialize the perturbation as zeros
        model = model_mod.get_model()
        image_m = model_mod.preprocess_image(image)
        image_f = FGSM_mod.FGSM_preprocess(image)
        delta = zeros(eltype(image_m), size(image_m))
        grads_m = zeros(eltype(delta), size(delta))
    
        for _ in 1:max_iterations
            # Define the CW loss function to optimize
            loss = cw_loss(delta, image_m, target_class, model, c)
    
            # Compute gradients of the loss with respect to delta
            gs = gradient(() -> loss, params([delta]))
            gs .+= IdDict(p => randn(size(p)) for p in keys(gs))
            
            for (p, g) in pairs(gs)
                grads_m += g 
            end
    
            # Update delta using gradient ascent (maximize the loss)
            delta .= delta .+ 0.0001 .* grads_m
    
            # Clip delta to ensure it stays within the epsilon constraint
            delta .= max.(-0.2, min.(delta, 0.2))
        end
    
        # Generate the adversarial example by applying the perturbation
        adversarial_example = image_f .+ delta

        # Clip the adversarial example to stay within [0, 1] range
        adversarial_example .=max.(0.0, min.(adversarial_example, 1.0))
        f_adv = reshape(adversarial_example, 224, 224, 3);
        visualise_CW(f_adv, delta)

        return f_adv, delta
    end
    
    function visualise_CW(adv_x, noise)
        reshaped_adv_x = reshape(adv_x, 3, 224, 224)
        a = colorview(RGB, reshaped_adv_x)
        n_reshaped = reshape(noise, 3, 224, 224)
        n = colorview(RGB, n_reshaped)
        save("CW_attack.jpg", a)
        save("CW_noise.jpg", n)
    end
    
end
