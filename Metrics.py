# Metrics
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K # type: ignore

smooth = 1e-15
#Dice_coefficient
def dice_coeff(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


#Dice loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)

#precision
def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(y_true * y_pred))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

#recall
def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(y_true * y_pred))
    possible_positives = tf.reduce_sum(tf.round(y_true))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

#f1 score
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + tf.keras.backend.epsilon())
#iou
def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return intersection / (union + 1e-6)