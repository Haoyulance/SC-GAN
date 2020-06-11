from src.utils.util import *
from src.net import SCGAN
from skimage.measure import compare_ssim
import os
import shutil
import nibabel as nib
import copy


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('g_reg', 0.001, 'generator regularizer')
flags.DEFINE_float('d_reg', 0.001, 'discriminator regularizer')
flags.DEFINE_boolean('floss', True, 'if using feature matching loss')
flags.DEFINE_boolean('reg', True, 'if using regularization')
flags.DEFINE_boolean('attn', True, 'if using self attention')
flags.DEFINE_integer('gpu', None, 'the ID of gpu to use')
flags.DEFINE_integer('training_size', 260, 'training data size')
flags.DEFINE_integer('epoches', 120, 'number of iteration')
flags.DEFINE_integer('l1_weight', 200, 'l1 weight')
flags.DEFINE_integer('B_weight', 200, 'B-rmse weight')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_string('data_dir', None, 'training data directory')
flags.DEFINE_string('modalities', 'flair_t1w_av45.masked', 'modalities to use in the training')
flags.DEFINE_string('logdir', None, 'path of tensorboard log')

if FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def main(args):
    root = os.path.join(FLAGS.data_dir,'train')
    train_set = np.array(os.listdir(root))
    val_root = os.path.join(FLAGS.data_dir,'val')
    val_set = np.array(os.listdir(val_root))
    modalities = FLAGS.modalities.split('_')
    num_samples = FLAGS.training_size

    d_flag = 1

    with tf.device('/gpu:%s' %FLAGS.gpu):
        model = SCGAN(FLAGS.img_size, FLAGS.img_size, FLAGS.img_size, ichan=len(modalities)-1, ochan=1,
                      l1_weight=FLAGS.l1_weight, B_weight=FLAGS.B_weight, floss=FLAGS.floss, lr=FLAGS.lr,
                      reg=FLAGS.reg, attn=FLAGS.attn, g_reg=FLAGS.g_reg, d_reg=FLAGS.d_reg)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        dir = FLAGS.logdir
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        f_summary = tf.summary.FileWriter(logdir=dir, graph=sess.graph)
        previous_rmse = 100
        for step in range(FLAGS.epoches*num_samples):
            pos = step % num_samples
            if pos == 0:
                np.random.shuffle(train_set)
                avg_psnr = []
                avg_ssim = []
                avg_rmse = []
                for val_subject in val_set:
                    val_list = list()
                    for num_modalities in modalities[:-1]:
                        A_test = os.path.join(val_root, val_subject, '%s.nii.gz' % num_modalities)
                        A_test = nib.load(A_test).get_fdata()
                        A_test = (A_test - np.mean(A_test[A_test != np.min(A_test)])) / np.std(A_test[A_test != np.min(A_test)])
                        A_test = (A_test - np.min(A_test)) / (np.max(A_test) - np.min(A_test))
                        A_test = np.expand_dims(A_test, axis=0)
                        A_test = np.expand_dims(A_test, axis=-1)
                        val_list.append(A_test)
                    test_input = np.concatenate(val_list, axis=4)
                    test_pet = os.path.join(val_root, val_subject, '%s.nii.gz'%modalities[-1])
                    test_pet = nib.load(test_pet)
                    test_pet = test_pet.get_fdata()
                    test_pet = (test_pet - np.mean(test_pet[test_pet != np.min(test_pet)])) / np.std(test_pet[test_pet != np.min(test_pet)])
                    test_pet = (test_pet - np.min(test_pet)) / (np.max(test_pet) - np.min(test_pet))
                    temp_pet = test_pet * 2 - 1

                    output = model.sample_generator(sess, test_input, is_training=False)[0, :, :, :, 0]

                    mask, masked_temp_pet, masked_output= generate_mask(temp_pet, output)
                    dif = (masked_output - masked_temp_pet)**2
                    mse_raw = []
                    mse2_raw = []
                    mse3_raw = []
                    for po in range(FLAGS.img_size):
                        mse_raw.append(dif[po,:,:][mask[po,:,:]])
                        mse2_raw.append(dif[:,po,:][mask[:,po,:]])
                        mse3_raw.append(dif[:,:,po][mask[:,:,po]])

                    mse = get_mean(mse_raw)
                    mse2 = get_mean(mse2_raw)
                    mse3 = get_mean(mse3_raw)
                    rmse = (np.mean(np.sqrt(mse)) + np.mean(np.sqrt(mse2)) + np.mean(np.sqrt(mse3)))/3/2
                    mse = get_psnr_mean(mse_raw, FLAGS.img_size)
                    mse2 = get_psnr_mean(mse2_raw, FLAGS.img_size)
                    mse3 = get_psnr_mean(mse3_raw, FLAGS.img_size)
                    mse[mse==0] = 1e-10
                    mse2[mse2 == 0] = 1e-10
                    mse3[mse3 == 0] = 1e-10
                    print(val_subject)
                    print('nrmse = %s' % rmse)
                    psnr = np.mean(10 * np.log10(4 / mse))
                    psnr2 = np.mean(10 * np.log10(4 / mse2))
                    psnr3 = np.mean(10 * np.log10(4 / mse3))
                    psnr = np.mean([psnr, psnr2, psnr3])
                    print('psnr = %s' % psnr)
                    ssim = []
                    for image in range(output.shape[0]):
                        ssim.append(compare_ssim(masked_output[image, :, :], masked_temp_pet[image, :, :], data_range=2, win_size=11,
                                                 gaussian_weights=True))
                    ssim = np.mean(ssim)
                    print('ssim = %s' % ssim)
                    avg_psnr.append(psnr)
                    avg_rmse.append(rmse)
                    avg_ssim.append(ssim)

                psnr = np.mean(avg_psnr)
                ssim = np.mean(avg_ssim)
                rmse = np.mean(avg_rmse)

                if rmse < previous_rmse:
                    saver.save(sess, os.path.join(FLAGS.logdir, 'model_step_%s'%step))
                    previous_rmse = rmse
                    print("saved step %s model" % step)
                    print("current best rmse%s"%rmse)

                merge = tf.summary.merge([model.psnr, model.ssim, model.rmse])
                summary = sess.run(merge, feed_dict={model.test_psnr: psnr, model.test_ssim: ssim, model.test_rmse: rmse})
                f_summary.add_summary(summary=summary, global_step=step)

            subject = train_set[pos]
            try:
                input_list = list()
                for num_modalities in modalities[:-1]:
                    a = os.path.join(root, subject, '%s.nii.gz' %num_modalities)
                    a = nib.load(a).get_fdata()
                    a = (a - np.mean(a[a != np.min(a)])) / np.std(a[a != np.min(a)])
                    a = (a - np.min(a)) / (np.max(a) - np.min(a))
                    a = np.expand_dims(a, axis=0)
                    a = np.expand_dims(a, axis=-1)
                    input_list.append(a)
                input = np.concatenate(input_list, axis=4)
                pet = os.path.join(root, subject, '%s.nii.gz'%modalities[-1])
                pet = nib.load(pet).get_fdata()
                pet = (pet - np.mean(pet[pet != np.min(pet)])) / np.std(pet[pet != np.min(pet)])
                pet = (pet - np.min(pet)) / (np.max(pet) - np.min(pet))
                pet = np.expand_dims(pet, axis=0)
                pet = np.expand_dims(pet, axis=-1)
            except:
                continue

            model.lr = cosine_decay(learning_rate=FLAGS.lr, global_step=step, decay_steps=num_samples*10, alpha=1e-10)
            pet = 2 * pet -1
            gloss_curr, dloss_curr, summary_temp = model.train_step(sess, input, input, pet, train_d=d_flag)
            f_summary.add_summary(summary=summary_temp, global_step=step)
            if d_flag == 1:
                print('Step %d: generator loss: %f | discriminator loss1: %f' % (step, gloss_curr, dloss_curr) + ' | ' + subject)
            else:
                print('Step %d: generator loss: %f' % (step, gloss_curr))



if __name__ == "__main__":
    tf.app.run()
