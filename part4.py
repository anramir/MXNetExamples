import mxnet as mx
import cv2
import numpy as np
import time

'''
Implementation of 
https://towardsdatascience.com/an-introduction-to-the-mxnet-api-part-4-df22560b83fe
and
https://medium.com/@julsimon/an-introduction-to-the-mxnet-api-part-5-9e78534096db
'''


def load_model(filename):
    sym, arg_params, aux_params = mx.model.load_checkpoint(filename, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    mod.set_params(arg_params, aux_params)
    return mod


def load_categories():
    synsetfile = open('synset.txt', 'r')
    synsets = []
    for l in synsetfile:
        synsets.append(l.rstrip())
    return synsets


def read_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224, ))
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    image = image[np.newaxis, :]
    return mx.nd.array(image)


def predict(image, model):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    input = Batch([image])
    t1 = time.time()
    model.forward(input)
    t2 = time.time()
    print("Predicted in %2.8f milliseconds" % (t2 - t1))
    return np.squeeze(model.get_outputs()[0].asnumpy())


def top_n(prediction, categories, n):
    indexes = np.argsort(prediction)[::-1]
    return [(prediction[index], categories[index]) for index in indexes[0:n]]


def print_predictions(predictions):
    for (prob, category) in predictions:
        print("\tProb: %s,\tCategory: %s" % (prob, category))


def classify_image_with_model(model_name, image_name):

    model = load_model('models/%s' % model_name)
    image = read_image('images/%s' % image_name)
    categories = load_categories()
    print("Model: %s, Image: %s" % (model_name, image_name))
    prediction = predict(image, model)
    top_n_categories = top_n(prediction, categories, 5)
    print_predictions(top_n_categories)


if __name__ == '__main__':
    classify_image_with_model('Inception-BN', 'Captura.jpg')
    # classify_image_with_model('vgg16', 'Captura.jpg')
    classify_image_with_model('resnet-152', 'Captura.jpg')




