module CW_mod
    using Images
    using Colors
    using Flux
    using Flux: params, update!, Optimise, setup, onehotbatch, logitcrossentropy
    include("model.jl")
    include("one_pixel.jl")
    using .onepix_mod

    import .model_mod
    export CW_attack

    # Precompute the model once outside the loss function to avoid redundant computations
    model = model_mod.get_model()

    function cw_loss(delta, x, t, c, λ)
        # Compute logits for the original input
        original_logits = model(Flux.unsqueeze(x, 4))

        # Compute logits for the adversarial input with the perturbation delta
        adversarial_logits = model(Flux.unsqueeze(x, 4) .+ delta)

        # Calculate the Euclidean distance between the logits efficiently without using sum and sqrt
        euclidean_dist = sqrt(sum((adversarial_logits .- original_logits).^2)) + eps()

        # Calculate the L2 regularization term
        l2_term = λ * sum(delta.^2)

        # Calculate the cross-entropy loss between the adversarial logits and the target class
        target_onehot = onehotbatch([t], 1:size(adversarial_logits, 1))
        ce_loss = logitcrossentropy(adversarial_logits, target_onehot)

        # Combine the Euclidean distance, cross-entropy loss, and L2 regularization with a margin constraint
        loss = euclidean_dist + c * max(0.0, ce_loss - 0.5) + l2_term

        return loss
    end

    function CW_attack(image, target_class, c, δ, max_iterations, λ)
    
        # Initialize the perturbation with random values
        img = model_mod.preprocess_image(image)
        image_p = onepix_mod.normalize_data(img)
        initial_perturbation = 0.001 * δ * randn(size(image_p))
        delta = 0.001 * (initial_perturbation .- minimum(initial_perturbation)) / (maximum(initial_perturbation) - minimum(initial_perturbation))
        adversarial_example = image_p .+ delta
        adv_label = model_mod.predict(adversarial_example)
        grads_m = zeros(eltype(delta), size(delta))
    
        # Define the initial step size
        step_size = 0.001
    
        for iteration in 1:max_iterations
            # Define the CW loss function to optimize
            loss = cw_loss(delta, image_p, target_class, c, λ)
    
            # Compute gradients of the loss with respect to delta
            gs = gradient(() -> loss, params([delta]))
            gs .+= IdDict(p => randn(size(p)) for p in keys(gs))
    
            for (p, g) in pairs(gs)
                grads_m += g
            end
    
            # Calculate an adaptive step size (learning rate annealing)
            adaptive_step = step_size / (sqrt(iteration) + 1e-3)
    
            # Gradient descent update with adaptive step size
            delta = delta .+ adaptive_step .* grads_m
    
            # Clip delta to stay within the (0,1) range
            delta = clamp.(delta, 0, 1)
    
            # Visualize and return the adversarial example if found
            adv_example = image_p .+ delta
            adv_label = model_mod.predict(adv_example)
    
            # Save the visualization at each iteration
            visualise_CW(adv_example, delta, iteration)
    
            println("Iteration: $iteration, Loss: $loss, Adversarial Label: $(adv_label[2])")
    
            if adv_label[2] != target_class
                visualise_CW(adv_example, delta, "final")  # Save the final visualization
                return adv_example, delta
            end
        end
    
        # If no misclassification found after max_iterations, return a message
        println("Couldn't find an adversarial example leading to misclassification.")
        return adversarial_example, delta
    end
    
    function visualise_CW(adv_x, noise, iteration)
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
    
        # Save the visualization with an iteration number or "final" for the final image
        filename = iteration == "final" ? "CW_attack_final.jpg" : "visualizations_CW/CW_attack_$iteration.jpg"
        save(filename, img)
        save("visualizations_CW/CW_noise_$iteration.jpg", n)
    end
end
