# testing CNN


import fruits_data
import tensorflow as tf
tf.set_random_seed(0)
import numpy as np
import cnn
import os
import plot_utils
import glob
import math
from MyFunctions import burst_error_array
from MyFunctions import save_statistics
import argparse

tf.reset_default_graph()

DATASETS = {
    "fruits": {
        "image_size": 100,
        "class": fruits_data,
        "channels": 3,
        "order": 'F'
    }
}

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--filter_size', type=int, default='128', help='Convolutional Layer Filter Size',
                        required=False)


    parser.add_argument('--kappa', type=float, default='10', help='significance of image similiarity for total loss',
                        required=False)

    parser.add_argument('--restore', type=bool, default=False, help='restore saved training',
                        required=False)

    parser.add_argument('--save', type=bool, default=False, help='save training',
                        required=False)

    parser.add_argument('--no_KL', type=int, default=1, help='no KL divergence if 0',
                        required=False)

    parser.add_argument('--savefile', type=str, default='', help='name of savefile',
                        required=False)

    parser.add_argument('--keep_prob', type=float, default='1', help='Keep probability for dropout',
                        required=False)

    parser.add_argument('--test_SNRs', type=int, default='1', help='if True, test model for different SNRs',
                        required=False)

    parser.add_argument('--burst_error', type=int, default='0', help='if 1, test for burst error',
                        required=False)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--deterministic', type=int, default='0', required=False,
                        help='If 1, then use a deterministic network')

    parser.add_argument('--dataset', type=str, default='fruits',
                        help='Choose image dataset. Options: {}'.format(DATASETS.keys()))

    parser.add_argument('--channel_noise_var', type=float, default=0.15, help='Variance of the channel noise')

    parser.add_argument('--dim_z', type=int, default='5', help='Dimension of latent vector', required=False)

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=4000, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')

    parser.add_argument('--PRR', action='store_true', default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=32,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=18,
                        help='Number of images along y-axis')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path + '/*')
    for f in files:
        os.remove(f)

    try:
        assert args.dataset in DATASETS.keys()
    except:
        print('Dataset not recognized. Available options: {}'.format(DATASETS.keys()))
        return None

    # --channel_noise_var
    try:
        assert args.channel_noise_var > 0
    except:
        print('channel_noise_var must be positive number')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')


    return args


"""main function"""


def main(args):
    RESULTS_DIR = args.results_path


    #Load arguments
    dataset = DATASETS[args.dataset]
    channel_std = np.sqrt(args.channel_noise_var)  # only use these values case train_multiple_SNR is False
    res = dataset["image_size"]
    channels = dataset["channels"]
    filter_size = args.filter_size
    dim_z = args.dim_z
    kappa=args.kappa
    batch_size = args.batch_size
    n_epochs = args.num_epochs
    learn_rate = args.learn_rate
    PRR_n_img_x = args.PRR_n_img_x  # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y  # number of images along y-axis in a canvas

    savestring = '_z%d_f%d_bs%d_kappa%d_KL%d_DET%d_channel%.2f'%(dim_z,filter_size,batch_size,args.kappa,args.no_KL,args.deterministic,args.channel_noise_var)
    RESULTS_DIR = RESULTS_DIR + savestring
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(RESULTS_DIR + '/saves'):
        os.makedirs(RESULTS_DIR + '/saves')


    #Load Datasets
    test_tot_imgs = PRR_n_img_x * PRR_n_img_y
    traind, test_data= dataset["class"].prepare_data(test_tot_imgs,dataset["order"])
    n_samples = traind.shape[0]

    # Plot for reproduce performance
    PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y)
   # PRR.save_images(test_data, name='input.jpg')



    #Save parameters in text
    with open("%s/parameters.txt" % (RESULTS_DIR), "w") as text_file:
        print("filter: {}".format(filter_size), file=text_file)
        print("dim_z: {}".format(dim_z), file=text_file)
        print("learning_rate: {}".format(learn_rate), file=text_file)
        print("batch_size: {}".format(batch_size), file=text_file)
        print("Channel_noise_var: {}".format(args.channel_noise_var), file=text_file)
        print("epochs: {}".format(n_epochs), file=text_file)
        print("kappa neo: {}".format(args.kappa), file=text_file)
        print("KL-term weight: {}".format(args.no_KL), file=text_file)
        print("Deterministic: {}".format(args.deterministic), file=text_file)

    """ build graph """
    # input placeholders


    x_hat = tf.placeholder(tf.float32, shape=[None, res, res, channels], name='target_img')

    channel_noise_stddev = tf.placeholder(tf.float32, shape=(), name="channel_noise_stddev")

    burst_noise = tf.placeholder(tf.float32,shape=(1,dim_z),name="burst_noise")


    y, loss, mse, KL_divergence, piesenar,mu,sigma = cnn.autoencoder(x_hat, dim_z,
        channel_noise_stddev,O1=filter_size, kapa=kappa,no_KL=args.no_KL, deterministic=args.deterministic,
                                                                            burst_error=burst_noise)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate,epsilon=1e-6).minimize(loss)



    """ training """





#INITIALIZE ARRAYS AND VARIABLES
    total_batch = int(n_samples / batch_size)
    #initiliaze vectors to calculate the mean of all batches in each epoch
    PSNR_train = np.zeros((total_batch))
    loss_divergence_batch=np.zeros((total_batch))
    tot_loss_batch=np.zeros((total_batch))
    MSE_batch = np.zeros((total_batch))

    #Initiliaze Array with all results
    Results = np.zeros((args.num_epochs, 7))

    #initiliaze output test array and test PSNR batch array
    y_PRR = np.zeros(test_data.shape)
    PSNR_ = np.zeros((test_tot_imgs))

    best_PSNR = -1

    #save filename extension based on parameters
    savestring = '_z%d_f%d_bs%d_kappa%d_KL%d'%(dim_z,filter_size,batch_size,kappa,args.no_KL)

    # function to convert snr to stddev
    snr2stddev = lambda snr: math.sqrt(1 / (10 ** (snr / 10)))

    if args.burst_error==1:burst_array = burst_error_array(0.98, 0.3, dim_z)
    zero_array=np.zeros((1,dim_z))




    #settings for session
    saver = tf.train.Saver()
    tf.add_to_collection('train', train_op)
    config = tf.ConfigProto()
    #run_options=tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess:

        # initialize or restore the model
        if args.restore == True:

            new_saver = tf.train.import_meta_graph('%s/saves/my_model%s.meta' % (RESULTS_DIR,savestring))
            new_saver.restore(sess, '%s/saves/my_model%s' % (RESULTS_DIR,savestring))
            train_op = tf.get_collection('train_op')[0]
        else:
            sess.run(tf.global_variables_initializer(),
                     feed_dict={channel_noise_stddev: channel_std})#,options=run_options)

        for epoch in range(n_epochs):


            np.random.seed(epoch)
            np.random.shuffle(traind)
            train_data_=traind


            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]

                yhat, tot_loss_batch[i], MSE_batch[i], loss_divergence_batch[i], PSNR_train[i] = sess.run(
                    (train_op, loss,mse, KL_divergence, piesenar),
                    feed_dict={x_hat: batch_xs_input, channel_noise_stddev: channel_std, burst_noise:zero_array})

            Results[epoch, 0] = np.mean(PSNR_train)
            Results[epoch,3] = np.mean(MSE_batch)
            Results[epoch,5] = np.mean(loss_divergence_batch)
            Results[epoch,6] = np.mean(tot_loss_batch)
            total_loss= Results[epoch,6]

            print("epoch %d: L_tot %03.2f MSE %03.2f KL_divergence %03.2f PSNR %03.2f" % (
            epoch, total_loss, Results[epoch,3], Results[epoch,5], Results[epoch, 0]))

            # Testing
            K = batch_size #Test each K sample iteratively to avoid OOM
            similarity_ = np.zeros(int(test_data.shape[0] / K))

            for i in range(int(test_data.shape[0] / K)):
                y_PRR[K * i:K * (i + 1), :, :, :], PSNR_[i], similarity_[i],muu,sigmaa = sess.run((y, piesenar,
                                mse,mu,sigma),feed_dict={x_hat: test_data[K * i:K * (i + 1)],
                                       channel_noise_stddev: channel_std,burst_noise:zero_array})
            PSNR_=K*PSNR_
            Results[epoch, 1] = np.mean(PSNR_)
            Results[epoch, 4] = np.mean(similarity_)
            similarity = Results[epoch, 4]


            #save mean,std pair in string array
            Distributions=save_statistics(muu, sigmaa, deterministic=args.deterministic)



            print("TEST  PSNR: %03.2f   Similarity: %03.2f" % (Results[epoch, 1], Results[epoch, 4]))



            if Results[epoch, 1] > best_PSNR:

                best_PSNR = Results[epoch, 1]
                Results[epoch, 2] = 1
                np.savetxt("%s/distributions.out"%(RESULTS_DIR), Distributions, delimiter=",", newline="\n", fmt="%s")
                PRR.save_images(y_PRR, name='/EPOCH_%02d_PSNR%03.2f_MSE%03.0f_LOSS%01.3f_KL%1.3f.jpg' % (
                epoch, Results[epoch, 1], similarity, Results[epoch,3], Results[epoch,5]))
                print('IMPROVEMENT')

                if args.save == True: saver.save(sess, '%s/saves/my_model%s' % (RESULTS_DIR, savestring))




        np.savetxt("%s/PSNRs%s.out" % (RESULTS_DIR,savestring), Results, fmt='%02.2f %02.2f %1.d %02.2f %02.2f %.2f %.2f', delimiter='  ,  ')
        np.save("%s/PSNRs_array%s" % (RESULTS_DIR,savestring), Results)


        # Testing for multiple SNR's

        if args.test_SNRs == 1:
            fname='%s/saves/my_model%s'% (RESULTS_DIR, savestring)
            if os.path.isfile(fname+'.meta')==1:
                new_saver = tf.train.import_meta_graph(fname+'.meta')
                new_saver.restore(sess, fname)
                train_op = tf.get_collection('train_op')[0]

            M = 25  # max SNR tested
            TEST_SNRs = np.zeros((M, 4))

            TEST_SNRs_temp = np.zeros(int(test_data.shape[0]))
            MSE_test_SNRs_temp = np.zeros(int(test_data.shape[0]))
            for k in range(M):
                K = batch_size
                for i in range(int(test_data.shape[0] / K)):
                    np.random.seed(i)

                    if args.burst_error==1:
                        g=np.random.choice(burst_array.shape[0],1)
                        bERROR=burst_array[g,:]
                    else:
                        bERROR=zero_array

                    MSE_test_SNRs_temp[i],TEST_SNRs_temp[i] = sess.run((mse,piesenar), feed_dict={x_hat: test_data[K * i:K * (i + 1)],
                                                                     channel_noise_stddev: snr2stddev(k),burst_noise:bERROR})
                TEST_SNRs_temp=K*TEST_SNRs_temp
                TEST_SNRs[k, 1] = np.mean(TEST_SNRs_temp)
                TEST_SNRs[k, 0] = k
                TEST_SNRs[k, 2] = snr2stddev(k)

                MSE_test_SNRs_temp=K*MSE_test_SNRs_temp
                TEST_SNRs[k,3]=np.mean(MSE_test_SNRs_temp)



            np.save("%s/PSNR_test_SNR%s" % (RESULTS_DIR,savestring), TEST_SNRs)
            np.savetxt("%s/TEST_SNRs%s.out" % (RESULTS_DIR,savestring), TEST_SNRs, fmt='%d %02.2f %02.2f %02.2f', delimiter=',')


    sess.close()
    return best_PSNR

if __name__ == '__main__':

    # parse arguments

    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
