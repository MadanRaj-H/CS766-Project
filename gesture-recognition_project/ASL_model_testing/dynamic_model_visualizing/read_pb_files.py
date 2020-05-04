import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)
  return graph


graph = load_graph('models/retrained_graph_current_use.pb')
with tf.Session(graph=graph) as sess:
  model_filename = 'models/retrained_graph_current_use.pb'
  with gfile.GFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def)
LOGDIR = './logfiles'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
