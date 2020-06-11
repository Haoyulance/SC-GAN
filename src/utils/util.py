import tensorflow as tf
import numpy as np

def get_mean(matrix):
    result = list()
    for i in matrix:
        if not i.size:
            continue
        else:
            result.append(sum(i)/len(i))
    return np.array(result)

def get_psnr_mean(matrix, size):
    result = list()
    for i in matrix:
        if not i.size:
            continue
        else:
            # result.append(np.mean(i))
            result.append(np.sum(i)/size**2)
    return np.array(result)

def generate_mask(temp_pet, output):
    mask = np.ma.masked_where(temp_pet == -1, temp_pet)
    mask = np.ma.getmask(mask)
    masked_temp_pet = temp_pet
    masked_output = copy.deepcopy(output)
    masked_output[mask] = -1
    mask = np.ma.masked_where(temp_pet != -1, temp_pet)
    mask = np.ma.getmask(mask)
    return mask, masked_temp_pet, masked_output

def cosine_decay(learning_rate, global_step, decay_steps, alpha=0):
    # global_step = global_step % decay_steps
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed

    return decayed_learning_rate



def get_shape(tensor):
    return tensor.get_shape().as_list()

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)

def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)

def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = get_shape(weights)
    w_mat = tf.reshape(weights, [-1, w_shape[-1]]) # convert to 2 dimension but total dimension still the same
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /=sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def snconv3d(input_, output_dim, size, stride, sn_iters=1, update_collection=None):
    w = tf.get_variable('filter', [size, size, size, get_shape(input_)[-1], output_dim])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv

def snconv3d_tranpose(input_, output_dim_from, size, stride, sn_iters=1, update_collection=None):
    w = tf.get_variable('filter', [size, size, size, get_shape(output_dim_from)[-1], get_shape(input_)[-1]])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d_transpose(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME',
                                  output_shape=tf.shape(output_dim_from))
    return conv

def snconv3d_1x1(input_, output_dim, sn_iters=1, sn=True, update_collection=None, init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
    with tf.variable_scope(name):
        w = tf.get_variable('filter', [1, 1, 1, get_shape(input_)[-1], output_dim], initializer=init)
        if sn:
            w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        else:
            w_bar = w
        conv = tf.nn.conv3d(input_, w_bar, strides=[1, 1, 1, 1, 1], padding='SAME')
        return conv

def sn_attention(name, x, sn=True, final_layer=False, update_collection=None, as_loss=False):
    with tf.variable_scope(name):
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()

        if not batch_size:
            batch_size = 1
        location_num = height * width * depth

        if final_layer:
            downsampled = location_num//(64**3)
            stride = 64
        else:
            downsampled = location_num // 8
            stride = 2

        # theta path
        theta = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//8 or 4), update_collection=update_collection, name='theta')
        theta = tf.reshape(theta, [batch_size, location_num, (num_channels//8 or 4)])

        # phi path
        phi = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//8 or 4), update_collection=update_collection, name='phi')
        phi = tf.layers.max_pooling3d(inputs=phi, pool_size=[2, 2, 2], strides=stride)
        phi = tf.reshape(phi, [batch_size, downsampled, (num_channels//8 or 4)])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # g path
        g = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//2 or 16), update_collection=update_collection, name='g')
        g = tf.layers.max_pooling3d(inputs=g, pool_size=[2, 2, 2], strides=stride)
        g = tf.reshape(g, [batch_size, downsampled, (num_channels//2 or 16)])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, height, width, depth, (num_channels//2 or 16)])
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0), trainable=True)
        attn_g = snconv3d_1x1(attn_g, sn=sn, output_dim=num_channels, update_collection=update_collection, name='attn')
        if as_loss:
            return attn_g
        else:
            return (x + sigma * attn_g)/(1+sigma), sigma
