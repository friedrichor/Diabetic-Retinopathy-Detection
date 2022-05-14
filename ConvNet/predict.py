import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import convnext_tiny as create_model

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "./weights/best_model2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_path = str(ROOT / '../data/data_split/test') + '/'
    num_correct = 0
    num_test_data = 0
    num_0_1 = len(os.listdir(test_path + '0-1'))
    num_2_3 = len(os.listdir(test_path + '2-3'))
    TP, FN, FP, TN = 0, 0, 0, 0
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
            num_tmp = 0
            for img in img_list:
                num_tmp += 1
                if num_tmp == num_0_1:
                    print('----------------------------')
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

                print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())

                for i in range(len(predict)):
                    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                              predict[i].numpy()))
                    if predict[i].numpy() > 0.5 and class_indict[str(i)] == cls:
                        num_correct += 1
                if predict[0].numpy() > 0.5 and cls == '0-1':  # 本为0-1，判为0-1
                    TP += 1
                if predict[1].numpy() > 0.5 and cls == '0-1':  # 本为0-1，判为2-3
                    FN += 1
                    print('错了')
                if predict[0].numpy() > 0.5 and cls == '2-3':  # 本为2-3，判为0-1
                    FP += 1
                    print('错了')
                if predict[1].numpy() > 0.5 and cls == '2-3':  # 本为2-3，判为2-3
                    TN += 1

    print(num_test_data)
    print('corr =', num_correct / num_test_data)
    # TP = TP / num_0_1
    # FN = FN / num_0_1
    # FP = FP / num_2_3
    # TN = TN / num_2_3
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2*precision*recall / (precision+recall)
    print(TP, FN, FP, TP)
    print('准确率:', accuracy)
    print('精确率:', precision)
    print('召回率:', recall)
    print('F1 score:', F1)


if __name__ == '__main__':
    main()
