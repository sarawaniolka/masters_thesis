# module CW_mod
#     using Images
#     using Colors
#     using Flux
#     using Flux: params
#     include("model.jl")
#     include("FGSM.jl")

#     import .model_mod
#     import .FGSM_mod
#     export CW_attack

#     model = model_mod.get_model()

# # Define the CW loss function
#     function cw_loss(delta, x, t, model, c)
#         predictions = Flux.softmax(model(x .+ delta))
#         target_prob = predictions[t]

#         max_diff = -Inf
#         max_i = -1

#         for i in eachindex(predictions)
#             if i != t
#                 diff = predictions[i] - target_prob
#                 if diff > max_diff
#                     max_diff = diff
#                     max_i = i
#                 end
#             end
#         end

#         loss = max(max_diff - euclidean_distance(predictions, t) + c, 0.0)
#         return loss
#     end

#     # Define Euclidean distance calculation
#     function euclidean_distance(x, y)
#         # Calculate the Euclidean distance between x and y
#         return sqrt(sum((x .- y).^2))
#     end

#     # Define the CW attack function
#     function CW_attack(x, target, steps::Integer)
#         opt = Flux.ADAM()
#         x_p = FGSM_mod.FGSM_preprocess(x)
#         x_m = model_mod.preprocess_image(x)
#         delta = zero(x_p) .+ Float32(1E-5)
#         r = Float32(1.0)
#         for _ in 1:3 # for 3 different random initializations
#             delta = delta .* r
#             ps = params(delta)
#             for _ in 1:steps
#                 l = cw_loss(delta, x_m, target, model, 0.1)
#                 grads = gradient(() -> l, ps)
#                 for p in ps
#                     # Update each parameter p using the gradient information g[p]
#                     if p in keys(grads)
#                         update!(opt, p, grads[p])
#                     end
#                 end
#             end
#             r = maximum(delta)
#         end
#         return clamp.(x_p .+ delta, 0, 1)
#     end
    
# end
