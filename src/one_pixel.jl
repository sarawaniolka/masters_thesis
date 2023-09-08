module onepix_mod
    using Flux
    using Images
    using Random

    include("model.jl")
    include("FGSM.jl")
    import .model_mod
    import .FGSM_mod


    function normalize_data(img)
        # Scale the pixel values to the (0,1) range
        min_val = minimum(img)
        max_val = maximum(img)
        pixel_data = (img .- min_val) / (max_val - min_val)
        return pixel_data
    end


    function visualise_image(img, iteration)
        reshaped_adv_x = permutedims(img, [3, 1, 2])
        # Create an image from the normalized data
        img = colorview(RGB, reshaped_adv_x)
        save("attacks_visualised/one_pixel_attack_$iteration.jpg", img)
    end

    function one_pixel_attack(img, max_iterations, num_pixels_to_change)
        preprocessed_image = model_mod.preprocess_image(img);
        p = onepix_mod.normalize_data(preprocessed_image)
        original_prediction = model_mod.predict(p);

        for i in 1:max_iterations
            pixels = select_pixels(num_pixels_to_change)
            altered_picture = copy(p)

            for pixel in pixels
                width, height, channel = pixel
                altered_picture[width, height, channel] = 0.0
            end
            adversarial_prediction = model_mod.predict(altered_picture)

            if adversarial_prediction[2] != original_prediction[2] && adversarial_prediction[2] != 286
                visualise_image(altered_picture, i)
                println("Adversarial found in iteration ", i)
                return altered_picture, pixels
            else
                println("Unable to find an adversarial attack with $num_pixels_to_change pixels")
            end
        end
        return preprocessed_image
    end
    
    

    function select_pixels(num_pixels)
        pixels = [(rand(1:224), rand(1:224), rand(1:3)) for _ in 1:num_pixels]
        return pixels
    end
    
end
