import tensorflow as tf
import sys
import os

# Convert the model from frozen graph.
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='/home/mlst01/CarCarder/tensorflow_LPRnet/finalCKP/frozen_graph.pb',
                    # both `.pb` and `.pbtxt` files are accepted.
    input_arrays=['inputs'],
	input_shapes={'inputs' : [1, 24, 94, 3]},
    output_arrays=['decoded']
)

# converter = tf.lite.TFLiteConverter.from_saved_model("./finalCKP/0/", tags=["serve"])

# converter.post_training_quantize = True
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

tflite_model_name = os.path.join('./tflite/', sys.argv[1])
print(tflite_model_name)

# Save the model.
with open(tflite_model_name, 'wb') as f:
  f.write(tflite_model)
