import numpy as np
from scipy.interpolate import interp1d

class Note():
    '''
    Contains:
        - tone type
        - note start time
        - note length
        - note frequency (can be vector, indicates frequency changing over time, in Hz)
        - fs


    '''
    def __init__(self):
        pass
    def gen_sig(self):
        pass


class PathAutomator():
    '''
    Elements:
    Base Path
    amplitude automation function:
        input: time since start, output: envelope mutliplier
    phase automation
        input:
    reparameterization:
        input: time, desired_phase, output: desired t in parameterization
    '''


class Tone():
    '''
    Contains amplitude generation method.
    Elements:
    Base Path
    - amplitude response
    - TODO phase response
    - TODO reparamerization of path



    Methods:
        -Multiply by scalar
        -modulate by another tone (multiply)
        -add tones
        -add scalar
        -eval_coords

    '''
    def __init__(self,path_list,harmonics,env_amp=None,env_time=None):
        '''

        :param harmonics:
        :param env_amp:
        :param env_time:
        '''
        if env_amp is None:
            raise(ValueError('Env_amp must be set for now.'))
        (path_list,harmonics,env_amp,env_time),ndims = self.valid_inputs(path_list,harmonics,env_amp,env_time)
        self.ndims = ndims

        self.harmonics = np.atleast_1d(harmonics)
        n_harmonics = self.n_harmonics = len(harmonics)
        self.n_amp = np.shape(env_amp,-1)

        n_points_env = self.n_points_env = env_amp.shape[-1]
        self.env_amp = np.array(env_amp)*np.ones(n_harmonics,n_points_env)
        self.env_time = np.array(env_amp)*np.ones(n_harmonics,n_points_env)

        self._env_funcs = [interp1d(env_time[ind], env_amp[ind]) for ind in range(n_harmonics)]

    def update_interp_func(self,env_amp,env_time):
        env_amp = np.array(env_amp) * np.ones(self.n_harmonics, self.n_points_env)
        env_time = np.array(env_amp) * np.ones(self.n_harmonics, self.n_points_env)
        self._env_funcs = [interp1d(env_time[ind], env_amp[ind], fill_value=(np.nan, env_amp[ind, -1]))
                           for ind in range(self.n_harmonics)]


    def eval_coords(self,fs,des_t,start_time):
        des_t = np.asarray(des_t)
        eff_t = des_t - start_time
        eff_phase = eff_t*2*np.pi
        coords =np.zeros(eff_phase.shape +  self.ndims)
        for path,ind in enumerate(self.path_list):
            env = self._env_funcs[ind](eff_t)
            path_phase = eff_phase*(fs*self.harmonics[ind])
            path_coords = env* path.eval_coords(
                path_phase%1) #TODO: add in min_val, max_val
            coords+= path_coords
        return coords
            




    @staticmethod
    def valid_inputs(path_list,harmonics,env_amp,env_time):

        try:
            harmonics, env_amp, env_time = map(np.array, [harmonics, env_amp, env_time])
        except TypeError:
            raise (TypeError('Invalid type encountered on initialization'))

        harmonics = np.atleast_1d(harmonics)

        if len(path_list)==1:
            path_list = path_list*len(harmonics)
        assert(len(path_list)==len(harmonics))

        if env_amp.shape != env_time.shape:
            raise(ValueError('Inconsistent shapes of env_amp and env time. (Env_amp; env_time ).shape:({}; {})'.format(
                env_amp.shape,env_time.shape)))

        assert env_amp.shape[:-1]==harmonics.shape


        ndims_list = [path.ndims for path in path_list]
        if any(ndims!=ndims_list[0] for ndims in ndims_list):
            raise ValueError('Inconsistent dimensions encountered in paths.')

        ndims = ndims_list[0]
        return (path_list,harmonics,env_amp,env_time),ndims

