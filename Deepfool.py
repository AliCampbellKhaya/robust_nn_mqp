def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    #forward call without softmax start
    f_imagea = net.conv_layer1(image)
    f_imagea = net.conv_layer2(f_imagea)
    f_imagea = f_imagea.view(f_imagea.size(0), -1)
    f_imagea = net.fc_layer(f_imagea)
    #forward call without softmax end
    f_image = f_imagea.data.cpu().numpy().flatten()
    #f_image = net.forward(image).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    x = pert_image.clone().detach().requires_grad_(True)
    #fs = net.forward(x)
    #forward call without softmax start
    fs = net.conv_layer1(x)
    fs = net.conv_layer2(fs)
    fs = fs.view(fs.size(0), -1)
    fs = net.fc_layer(fs)
    #forward call without softmax end
    fs_list = [fs[0,k] for k in range(num_classes)]
    k_i = label
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        x.grad = None
        fs[0, 0].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()
        for k in range(1, num_classes):
            x.grad = None
            #x.zero_grad()
            fs[0, k].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, k] - fs[0, 0]).data.cpu().numpy()
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        x = pert_image.clone().detach().requires_grad_(True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        loop_i += 1
    r_tot = (1+overshoot)*r_tot
    return r_tot, loop_i, label, k_i, pert_image
