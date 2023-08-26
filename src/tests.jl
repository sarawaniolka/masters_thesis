# One Pixel Attack
module onepix_mod
    using Flux
    using Images
    #using Optim

    include("model.jl")
    include("FGSM.jl")
    import .model_mod
    import .FGSM_mod


    function perturbe_image(img, pixel)
        perturbed_image = copy(img)
        perturbed_image[pixel...] = 1.0
        a = reshape(perturbed_image, 3, 224, 224)
        b = colorview(RGB, a)
        save("one_pixel_attack.jpg", b)
    end

    function one_pixel_attack(img)
        initial_pixel = (224 ÷ 2, 224, 3, 1)
        preprocessed_image = FGSM_mod.FGSM_preprocess(img)
        perturbe_image(preprocessed_image, initial_pixel)
    end
end