import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # num_class need to change due to number of classifications
    model = create_model(num_classes=2, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-15.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # load image
    # img_path = "../tulip.jpg"
    test_path = './data/data_split/test/'

    num_correct = 0
    for cls in os.listdir(test_path):
        img_list = []
        for img_path in os.listdir(test_path + cls):
            img_path = test_path + cls + '/' + img_path
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            img_list.append(img)

        with torch.no_grad():
            # predict class
            for img in img_list:
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

                print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                             predict[predict_cla].numpy())
                plt.title(print_res)
                print(max(predict))
                for i in range(len(predict)):
                    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                              predict[i].numpy()))
                    if predict[i].numpy() > 0.5 and class_indict[str(i)] == cls:
                        num_correct += 1
    print('corr =', num_correct / 40)
    # plt.show()


if __name__ == '__main__':
    main()
