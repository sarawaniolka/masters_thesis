# One Pixel Attack
module onepix_mod
    using Flux
    using Images
    using Random
    #using Optim

    include("model.jl")
    include("FGSM.jl")
    import .model_mod
    import .FGSM_mod


    function perturb_image(img, pixel)
        perturbed_image = copy(img)
        perturbed_image[pixel...] = 1.0
        one_pixel_data = reshape(perturbed_image, 224, 224, 3)
        a = reshape(perturbed_image, 3, 224, 224)
        b = colorview(RGB, a)
        save("one_pixel_attack.jpg", b)
        return one_pixel_data
    end

    function one_pixel_attack(img)
        width = rand(1:224)
        height = rand(1:224)
        channel = rand(1:3)
        initial_pixel = (width, height, channel, 1)
        
        println("Randomized Initial Pixel: $initial_pixel")
        
        preprocessed_image = FGSM_mod.FGSM_preprocess(img)
        a = perturb_image(preprocessed_image, initial_pixel)
        return a
    end
    
end