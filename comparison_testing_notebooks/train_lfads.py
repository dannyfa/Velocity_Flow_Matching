import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import fire


sys.path.insert(0,'/hdd/miles/velocity_cfm/dnnlib')
from util import Vanderpol,DoubleCircles,DoubleSDE,\
    Lorenz63,Lorenz96,Rossler,projection,Balls,ToyDsetDynamics

sys.path.insert(0,'/hdd/miles/velocity_cfm/lfads')
import lfads
from run_lfads import train, flags, hps_dict_to_obj

VALID_DS = ['Vanderpol','DoubleCircles','DoubleSDE',\
            'Lorenz63','Lorenz96','Rossler','Balls']


class HiddenPrints:
    def __enter__(self):
        self._original_stdout=sys.stdout
        sys.stdout = open(os.devnull,'w')

    def __exit__(self,exc_type,exc_val,exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def make_lfads_dict(flags_dict,true_dim,data_dim,dist='gaussian',savedir=''):
    # output parameters
    flags_dict['output_dist'] = dist
    # generation parameters
    flags_dict['ic_dim'] = 100 #initial conditions dimension
    flags_dict['factors_dim'] = 32 # output of generator, prior to decoding
    flags_dict['ic_enc_dim'] = 64 #hidden dimension of encoding RNN

    flags_dict['gen_dim'] = 100 # hidden dimension of generator RNN

    flags_dict['ci_enc_dim'] = 64 # hidden size for encoder of control inputs
    flags_dict['con_dim'] =  64 # hidden size of controller
    
    flags_dict['batch_size']= 128
    flags_dict['lfads_save_dir'] = savedir

    flags_dict['l2_gen_scale'] = 2.25e-5
    flags_dict['l2_con_scale'] = 3.00e-4

    flags_dict['kl_ic_weight'] = 6.65e-7
    flags_dict['kl_co_weight'] = 5.00e-5
    return hps_dict_to_obj(flags_dict)

def z_score(data):
    mu,sd = np.nanmean(data,axis=(0,1),keepdims=True),np.nanstd(data,axis=(0,1),keepdims=True)
    return (data - mu)/sd

def train_lfads(dataset,datadir,modeldir,project_dim=3,scale=True,spikify=True):

    if dataset in VALID_DS:

        gen = eval(dataset + '()')

    else:
        print(f'please pick a dataset from {VALID_DS} (case-sensitive)')
        raise NotImplementedError
    
    print("generating train data....")
    trd = gen.generate(n=1248,T=1,dt=0.02,sigma=0.1)
    ted = gen.generate(n=312,T=1,dt=0.02,sigma=0.1)


    p = projection(origDim=trd[0].shape[-1],newDim=project_dim,\
                   projType='double swish',temp=1.5)
    projTrain,projTest=[p.project(t) for t in trd],[p.project(t) for t in ted]

    stackedTrain,stackedTest = np.stack(projTrain,axis=0),np.stack(projTest,axis=0)
    if scale:
        stackedTrain = z_score(stackedTrain)
        stackedTest = z_score(stackedTest)

    if spikify:
        pass

    
    gtTrainStacked,gtTestStacked =np.stack(trd,axis=0),np.stack(ted,axis=0)
    print('Done!')
    #with h5py.File(os.path.join(datadir,dataset+'_' + str(project_dim) +'dim.h5'),'w') as f:

    #    f.create_dataset('train_truth',data=gtTrainStacked)
    #    f.create_dataset('test_truth',data=gtTestStacked)
    #    f.create_dataset('train_projected',data=stackedTrain)
    #    f.create_dataset('test_projected',data=stackedTest)

    ds = {'train_data': stackedTrain,
               'valid_data': stackedTest,'data_dim':project_dim,
               'num_steps':stackedTrain.shape[1],
          'train_ext_input':np.zeros(stackedTrain.shape),
          'valid_ext_input':np.zeros(stackedTest.shape)}
    
    hps = make_lfads_dict(flags.FLAGS.flag_values_dict(),
                          true_dim=trd[0].shape[-1],
                          data_dim=projTrain[0].shape[-1],
                          savedir=modeldir)
    
    data = {dataset: ds}
    hps.kind = 'train'
    hps.dataset_names = []
    hps.dataset_dims = {}
    for key in data:
        hps.dataset_names.append(key)
        hps.dataset_dims[key] = data[key]['data_dim']
    hps.num_steps = list(data.values())[0]['num_steps']
    hps.ndatasets = len(hps.dataset_names)
    if hps.num_steps_for_gen_ic > hps.num_steps:
        hps.num_steps_for_gen_ic = hps.num_steps
    hps._clip_value = 200
    hps.learning_rate_stop = 0.001

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
    config.gpu_options.allow_growth=True

    sess = tf.compat.v1.Session(config=config)
    print("training or restoring model")
    with sess.as_default():
        with tf.device(hps.device):
                model = train(hps,data)

    print("getting data embeddings")
    with sess.as_default():
        ds = hps.dataset_names[0]
        with HiddenPrints():
            train_data_dict = model.eval_model_runs_avg_epoch(data_name=list(data.keys())[0],data_extxd=data[ds]['train_data'])
            #train_data_dict = model.eval_model_runs_push_mean(data_name=list(data.keys())[0],data_extxd=data[ds]['train_data'])
            test_data_dict = model.eval_model_runs_avg_epoch(data_name=list(data.keys())[0],data_extxd=data[ds]['valid_data'])
            #test_data_dict = model.eval_model_runs_push_mean(data_name=list(data.keys())[0],data_extxd=data[ds]['valid_data'])

    print("done! saving")
    scale_str = 'scaled' if scale else ''
    save_loc = os.path.join(datadir,dataset+ scale_str + '_' + str(project_dim) +'dim.h5')
    print(f"saving data at {save_loc}")
    with h5py.File(save_loc,'w') as f:

        f.create_dataset('train_truth',data=gtTrainStacked)
        f.create_dataset('test_truth',data=gtTestStacked)
        f.create_dataset('train_projected',data=stackedTrain)
        f.create_dataset('test_projected',data=stackedTest)
        f.create_dataset('train_embeddings',data=train_data_dict['factors'])
        f.create_dataset('test_embeddings',data=test_data_dict['factors'])
        f.create_dataset('train_recon_means',\
                         data=train_data_dict['output_dist_params'][:,:,:project_dim])
        f.create_dataset('test_recon_means',\
                         data=test_data_dict['output_dist_params'][:,:,:project_dim])

    print("all done!")
if __name__ == '__main__':

    fire.Fire(train_lfads)





