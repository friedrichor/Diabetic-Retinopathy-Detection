import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
from params import *


def main(args):
    device = args.device
    print('device =', device)

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = args.path_json
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    # num_class need to change due to number of classifications
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_path = str(args.path_test) + '/'

    num_correct = 0
    num_test_data = 0
    for cls in os.listdir(test_path):
        img_list = []
        num_test_data += len(os.listdir(test_path + cls))
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

    print('corr =', num_correct / num_test_data)
    # plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / model, help='model path(s)')  # 模型参数
    parser.add_argument('--path_test', type=str, default=ROOT / path_test, help='test datasets path')  # 测试集路径
    parser.add_argument('--path_json', type=str, default=ROOT / path_json, help='class_indice.json path')
    parser.add_argument('--num_classes', type=int, default=num_classes, help='number of classes')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    main(opt)
