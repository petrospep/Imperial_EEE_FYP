RUN main.py with input arguments as follow:

--filter_size', type=int, default='128', help='Convolutional Layer Filter Size',required=False


--kappa', type=float, default='10', help='regularization weight for KL divergence in the loss function',required=False



--restore', type=bool, default=False, help='restore saved training',required=False



--save', type=bool, default=False, help='save training',required=False



--no_KL', type=int, default=1, help='no KL divergence in loss function if 0',required=False



--test_SNRs', type=int, default='1', help='if True, test model for different SNRs',required=False


--results_path', type=str, default='results',help='File path of output'


--deterministic', type=int, default='0', required=False,help='If 1, then use a deterministic network'



--channel_noise_var', type=float, default=0.15, help='Variance of the channel noise'



--dim_z', type=int, default='5', help='Dimension of latent vector', required=False



--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer'



--num_epochs', type=int, default=4000, help='The number of epochs to train for



--batch_size', type=int, default=10, help='Batch size')



--PRR_n_img_x', type=int, default=32, help='Number of images along x-axis')
--PRR_n_img_y', type=int, default=18,help='Number of images along y-axis')
Last two arguments fix the number of images to visualize but also the number of the test set
