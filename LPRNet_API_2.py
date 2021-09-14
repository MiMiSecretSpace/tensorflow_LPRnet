import argparse, cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}

##############################
class LPRNet(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)
        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(
                    tf.float32,            
                    shape=(None, 24, 94, 3),
                    name='inputs')
            
            tf.import_graph_def(graph_def, {'inputs': self.input})

        #self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        
        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably. 
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph=self.graph)

    def test(self, data):

        # Know your output node name
        #input_tensor = self.graph.get_tensor_by_name('inputs:0')
        logits = self.graph.get_tensor_by_name('import/decoded:0')
        # "import/batch_normalization_1/gamma:0"
        
        output = self.sess.run(logits, feed_dict = {self.input: data})
        #print(output)

        return output
##########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frozen_graph_filename",
                        default="results/frozen_model.pb",
                        type=str,
                        help="Frozen model file to import")
    parser.add_argument("-i", "--test_image", 
                        default="test/image.jpg", 
                        type=str,
                        help="Image for test")
    args = parser.parse_args()

    tf.reset_default_graph()

    img = cv2.imread(args.test_image)
    print(img)
    img = cv2.resize(img, (94, 24))
    print(img)
    img_batch = np.expand_dims(img, axis=0)
    print("------------------------- img_batch ----------------------------")
    print(img_batch)

    with tf.io.gfile.GFile(args.frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    model = LPRNet(model_filepath=args.frozen_graph_filename)
    resultonehot = model.test(data=img_batch)
    print(resultonehot)
    print("----------------------------")
    #print(decode(resultonehot))
    decoded_labels = []
    for item in resultonehot:
        #print(item)
        expression = ['' if i == -1 else DECODE_DICT[i] for i in item]
        expression = ''.join(expression)
        decoded_labels.append(expression)

    for l in decoded_labels:
        print(l)
