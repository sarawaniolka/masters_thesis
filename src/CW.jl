module CW_mod
    using Flux
    using Flux: params
    include("model.jl")
    using Optim
    using Images
    using Colors
    import .model_mod

    model = model_mod.get_model();

    function custom_loss(x, y)
        return Flux.crossentropy(Flux.softmax(model(Flux.unsqueeze(x, 4))), y)
    end

    # Define the CW function
    function cw_attack(x, y; max_iterations=1000, initial_ϵ=0.01, model=model)
        img_array = channelview(x)
        img_array .= (img_array .- minimum(img_array)) / (maximum(img_array) - minimum(img_array))
        CW(img_array, y; max_iterations=1000, initial_ϵ=0.01)
    end
    
        

    function CW(x, y; max_iterations=1000, initial_ϵ=0.01)
        ϵ = initial_ϵ
        x = reshape(x, 224, 224, 3)
        perturbation = zeros(Float32, size(x))
        t=0
        # Iterate for a maximum number of iterations
        for i in 1:5
            # Compute the perturbed image
            perturbed_image = x + perturbation
            perturbed_image = clamp.(perturbed_image, 0.0, 1.0)

            # Compute the model
            t = model_mod.predict(perturbed_image)

            # Compute the loss
            loss = custom_loss(perturbed_image,t[2])

            # Compute the gradient of the loss w.r.t. the image
            grad = gradient(() -> custom_loss(perturbed_image, t[2]), params([perturbed_image]))
            # Update the perturbation using the gradient and a step size
            a =  float32(ϵ)* sign.(grad[perturbed_image])
            perturbation += a

            # Print progress
            @show i, loss

            # Early stopping condition if the loss goes to zero
            if loss < 1e-6
                break
            end
        end

        # Compute the adversarial image
        adversarial_image = clamp.(x + perturbation, 0.0, 1.0)

        visualise_CW(adversarial_image)

        return adversarial_image
    end

    function visualise_CW(adv_x)
        reshaped_adv_x = reshape(adv_x, 3, 224, 224)
        a = colorview(RGB, reshaped_adv_x)
        save("masters_thesis/CW_attack.jpg", a)
    end


end