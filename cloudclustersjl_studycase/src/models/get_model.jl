
function get_model(model_name)
    if model_name == "custom"
        include("./custom.jl")
        model = custom_model
        img_dims = (32,32)
    elseif model_name == "vgg16"
        include("./vgg16.jl")
        model = vgg16
        img_dims = (224, 224)
    elseif model_name == "resnet18"
        include("./resnet.jl")
        model = resnet18
        img_dims = (224, 224)
    elseif model_name == "mobilenetv3_small"
        include("./mobilenet.jl")
        model = mobilenetv3_small
        img_dims = (224, 224)
    elseif model_name == "mobilenetv3_large"
        include("./mobilenet.jl")
        model = mobilenetv3_large
        img_dims = (224, 224)
    end

    model, img_dims
end