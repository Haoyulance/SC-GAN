from src.utils.util import *
from skimage.measure import compare_ssim
import os
import nibabel as nib
import copy


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('gpu', 1, 'the ID of gpu to use')
flags.DEFINE_integer('test_size', 50, 'test data size')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_string('output', None, 'path of output folder')
flags.DEFINE_string('data_dir', None, 'test data directory')
flags.DEFINE_string('modalities', 'flair_t1w_av45.masked', 'modalities to use in the training')
flags.DEFINE_string('log_dir', None, 'path to the tensorboard log')

if FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '/gpu:%s'%FLAGS.gpu
modalities = FLAGS.modalities.split('_')

def main(args):
    log_list = os.listdir(FLAGS.log_dir)
    meta = []
    for name in log_list:
        if '.meta' in name:
            meta.append(name)
    meta.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    meta_graph = os.path.join(FLAGS.log_dir, meta[-1])
    meta_weight = os.path.join(FLAGS.log_dir, meta[-1].split('.')[0])

    saver = tf.train.import_meta_graph(meta_graph)
    sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True))
    with tf.device('/gpu:%s' % FLAGS.gpu):
        saver.restore(sess, meta_weight)
    graph = tf.get_default_graph()
    op = graph.get_tensor_by_name('Generator/decoder/final/tanh:0')
    root = os.path.join(FLAGS.data_dir,'test')
    num_subject = np.array(os.listdir(root))
    num_samples = FLAGS.test_size
    pa = FLAGS.output
    if not os.path.isdir(pa):
        os.makedirs(pa)
    avg_psnr = []
    avg_ssim = []
    avg_rmse = []
    for i in range(num_samples):
        input_list = list()
        for num_modalities in modalities[:-1]:
            A_test = os.path.join(root, num_subject[i], '%s.nii.gz'%num_modalities)
            A_test = nib.load(A_test).get_fdata()
            A_test = (A_test - np.mean(A_test[A_test != 0])) / np.std(A_test[A_test != 0])
            A_test = (A_test - np.min(A_test)) / (np.max(A_test) - np.min(A_test))
            A_test = np.expand_dims(A_test, axis=0)
            A_test = np.expand_dims(A_test, axis=-1)
            input_list.append(A_test)
        test_input = np.concatenate(input_list, axis=4)
        test_pet = os.path.join(root, num_subject[i], '%s.nii.gz'%modalities[-1])
        test_pet = nib.load(test_pet)
        affine = test_pet.affine
        test_pet = test_pet.get_fdata()
        test_pet = (test_pet - np.mean(test_pet[test_pet != 0])) / np.std(test_pet[test_pet != 0])
        test_pet = (test_pet - np.min(test_pet)) / (np.max(test_pet) - np.min(test_pet))
        temp_pet = test_pet * 2 - 1
        input = graph.get_tensor_by_name('input_holder:0')
        is_train = graph.get_tensor_by_name('is_train_holder:0')
        output = sess.run(op, feed_dict={input:test_input, is_train:False})[0, :, :, :, 0]
        tem = (output + 1) / 2
        mask = np.ma.masked_where(temp_pet == -1, temp_pet)
        mask = np.ma.getmask(mask)
        masked_temp_pet = temp_pet
        masked_output = copy.deepcopy(output)
        masked_output[mask] = -1
        mask = np.ma.masked_where(temp_pet != -1, temp_pet)
        mask = np.ma.getmask(mask)
        dif = (masked_output - masked_temp_pet) ** 2
        mse_raw = []
        mse2_raw = []
        mse3_raw = []
        for po in range(FLAGS.img_size):
            mse_raw.append(dif[po, :, :][mask[po, :, :]])
            mse2_raw.append(dif[:, po, :][mask[:, po, :]])
            mse3_raw.append(dif[:, :, po][mask[:, :, po]])
        mse = get_mean(mse_raw)
        mse2 = get_mean(mse2_raw)
        mse3 = get_mean(mse3_raw)
        rmse = (np.mean(np.sqrt(mse)) + np.mean(np.sqrt(mse2)) + np.mean(np.sqrt(mse3))) / 3 / 2
        mse = get_psnr_mean(mse_raw)
        mse2 = get_psnr_mean(mse2_raw)
        mse3 = get_psnr_mean(mse3_raw)
        mse[mse == 0] = 1e-10
        mse2[mse2 == 0] = 1e-10
        mse3[mse3 == 0] = 1e-10
        print(num_subject[i])
        print('nrmse = %s' % rmse)
        psnr = np.mean(10 * np.log10(4 / mse))
        psnr2 = np.mean(10 * np.log10(4 / mse2))
        psnr3 = np.mean(10 * np.log10(4 / mse3))
        psnr = np.mean([psnr, psnr2, psnr3])
        print('psnr = %s' % psnr)
        ssim = []
        for image in range(output.shape[0]):
            ssim.append(
                compare_ssim(masked_output[image, :, :], masked_temp_pet[image, :, :], data_range=2, win_size=11,
                             gaussian_weights=True))
        ssim = np.mean(ssim)
        print('ssim = %s' % ssim)
        avg_psnr.append(psnr)
        avg_rmse.append(rmse)
        avg_ssim.append(ssim)
        tem[tem < 1e-2] = 0
        dif = test_pet - tem
        new = nib.Nifti1Image(tem, affine)
        dif = nib.Nifti1Image(dif, affine)
        nib.save(dif, os.path.join(FLAGS.output, 'dif_%s.nii.gz'%num_subject[i]))
        nib.save(new, os.path.join(FLAGS.output, 'synthesis_%s.nii.gz'%num_subject[i]))

    psnr = np.mean(avg_psnr)
    ssim = np.mean(avg_ssim)
    rmse = np.mean(avg_rmse)
    print('average nrmse: %s'%rmse)
    print('average psnr: %s'%psnr)
    print('average ssim: %s'%ssim)

if __name__ == "__main__":
    tf.app.run()
