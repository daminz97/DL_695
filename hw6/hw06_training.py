from model import YoloNaive

def train_test(root_train, root_test, img_size, interval, path_saved, momentum, lr, epochs, bs, class_list, use_gpu, train_json=None, test_json=None, train=False, test=False):
    if train:
        yolo = YoloNaive(json_path= train_json,
                         root_train= root_train,
                         root_test = None,
                         img_size = img_size,
                         interval = interval,
                         path_saved_model = path_saved,
                         momentum = momentum,
                         learning_rate = lr,
                         epochs = epochs,
                         batch_size = bs,
                         class_list = class_list,
                         use_gpu = use_gpu
                         )
        print('\nLoading training data...\n')
        yolo.set_dataloaders(train=True)
        print('\nLoading training data completed\n')
        model = yolo.Net(skip_connections=True, depth=8)
        model = yolo.training_part(model, display_img=False)
    if test:
        yolo = YoloNaive(json_path= test_json,
                         root_train= None,
                         root_test = root_test,
                         img_size = img_size,
                         interval = interval,
                         path_saved_model = path_saved,
                         momentum = momentum,
                         learning_rate = lr,
                         epochs = epochs,
                         batch_size = bs,
                         class_list = class_list,
                         use_gpu = use_gpu
                         )
        print('\nLoading testing data...\n')
        yolo.set_dataloaders(test=True)
        print('\nLoading testing data completed\n')
        yolo.testing_part(model, display_img=True)


if __name__ == '__main__':
    train_json_path = 'annotations/instances_train2017.json'
    test_json_path = 'annotations/instances_val2017.json'
    root_train = 'coco_data/Train'
    root_test = 'coco_data/Val'
    img_size = 128
    interval = 20
    path_saved_model = 'results_data/saved_train_model.pt'
    momentum = 0.9
    learning_rate = 1e-5
    epochs = 20
    batch_size = 2
    class_list = ['chair', 'person', 'car']
    use_gpu = True
    train_test(root_train, root_test, img_size, interval, path_saved_model, momentum, learning_rate, epochs, batch_size, class_list, use_gpu, train_json=train_json_path, train=True)
    # train_test(root_train, root_test, img_size, interval, path_saved_model, momentum, learning_rate, epochs, batch_size, class_list, use_gpu, test_json=test_json_path, test=True)
