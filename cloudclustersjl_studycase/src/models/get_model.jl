
function get_model(model_name)
    if model_name == :custom
        include("src/models/custom.jl")
        model = custom_model
        img_dims = (32,32)
    elseif model_name == :vgg16
        include("src/models/vgg16.jl")
        model = vgg16
        img_dims = (224, 224)
    elseif model_name == :resnet18
        include("src/models/resnet.jl")
        model = resnet18
        img_dims = (224, 224)
    elseif model_name == :mobilenetv3_small
        include("src/models/mobilenet.jl")
        model = mobilenetv3_small
        img_dims = (224, 224)
    elseif model_name == :mobilenetv3_large
        include("src/models/mobilenet.jl")
        model = mobilenetv3_large
        img_dims = (224, 224)
    end

    model, img_dims
end