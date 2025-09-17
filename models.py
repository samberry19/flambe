import torch
import numpy as np 
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
from pyro.distributions import constraints

from .utils import *

### DEFINE BASE MODELS AND GUIDES ###

def linear(X, y=None, y_err=None, sigma_prior=0.1, beta_prior_scale=2, b_prior_loc=0., b_prior_scale=2.):
    """
    Bayesian model with sparse interactions, avoiding memory-intensive operations.
    """
    device = X.device; D = X.shape[1]

    # Priors on individual mutation effects
    beta = pyro.sample("beta", dist.Normal(torch.zeros(D, device=device), 
                               beta_prior_scale * torch.ones(D, device=device)).to_event(1))
    # Additional priors
    b = pyro.sample("b0", dist.Normal(torch.tensor(b_prior_loc, device=device), torch.tensor(b_prior_scale, device=device)))
    y_hat = X @ beta + b

    y_pred = gaussian_error_with_halfnormal_prior(y_hat, sigma_prior=sigma_prior, y_err=y_err, error_name="sigma_m")

    # Observed data likelihood
    with pyro.plate("data", X.shape[0]):  
        pyro.sample("obs", y_pred, obs=y)

def linear_guide(X, y=None, y_err=None):
    """
    Variational guide for sparse coupling model, only learning meaningful interactions.
    """
    device = X.device
    D = X.shape[1]

    # Variational parameters for beta
    beta_loc = pyro.param("beta_loc", torch.zeros(D, device=device))
    beta_scale = pyro.param("beta_scale", torch.ones(D, device=device) * 0.5, constraint=constraints.positive)

    # Variational parameters for b
    b_loc = pyro.param("b0_loc", torch.tensor(0., device=device))
    b_scale = pyro.param("b0_scale", torch.tensor(0.5, device=device), constraint=constraints.positive)
    
    # Variational posterior for sigma_extra
    sigma_m_loc = pyro.param("sigma_m_loc", torch.tensor(0.1, device=device), constraint=constraints.positive)
    sigma_m_scale = pyro.param("sigma_m_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)

    beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))
    b = pyro.sample("b0", dist.Normal(b_loc, b_scale))
    sigma_m = pyro.sample("sigma_m", dist.LogNormal(torch.log(sigma_m_loc), sigma_m_scale))

    return beta, b, sigma_m


def sigmoid(X, y=None, y_err=None, sigma_prior=0.1, beta_prior_scale=2, b_prior_loc=0., a_prior_scale=0.5, a_prior_loc=1, k_prior_scale=1, k_prior_loc=0, b_prior_scale=2.):
    """
    Bayesian model with sparse interactions, avoiding memory-intensive operations.
    """
    device = X.device
    D = X.shape[1]

    # Priors on individual mutation effects
    beta = pyro.sample("beta", dist.Normal(torch.zeros(D, device=device), 
                                           beta_prior_scale * torch.ones(D, device=device)).to_event(1))
    # Additional priors
    b0 = pyro.sample("b0", dist.Normal(torch.tensor(b_prior_loc, device=device), torch.tensor(b_prior_scale, device=device)))
    
    a = pyro.sample("a", dist.Normal(torch.tensor(a_prior_loc, device=device), torch.tensor(a_prior_scale, device=device)))
    k = pyro.sample("k", dist.Normal(torch.tensor(k_prior_loc, device=device), torch.tensor(k_prior_scale, device=device)))
    z = X @ beta + b0

    y_hat = a * torch.sigmoid(z.clamp(-10, 10)) + k  

    y_pred = gaussian_error_with_halfnormal_prior(y_hat, sigma_prior=sigma_prior, y_err=y_err, error_name="sigma_m")

    # Observed data likelihood
    with pyro.plate("data", X.shape[0]):  
        pyro.sample("obs", y_pred, obs=y)
        
def sigmoid_guide(X, y=None, y_err=None, sigma_prior=None):
    """
    Variational guide for sparse coupling model, only learning meaningful interactions.
    """
    device = X.device
    D = X.shape[1]

    # Variational parameters for beta
    beta_loc = pyro.param("beta_loc", torch.zeros(D, device=device))
    beta_scale = pyro.param("beta_scale", torch.ones(D, device=device) * 0.5, constraint=constraints.positive)

    # Variational parameters for b
    b0_loc = pyro.param("b0_loc", torch.tensor(0., device=device))
    b0_scale = pyro.param("b0_scale", torch.tensor(0.5, device=device), constraint=constraints.positive)

    # Variational parameters for a (kept positive)
    a_loc = pyro.param("a_loc", torch.tensor(1., device=device), constraint=dist.constraints.positive)
    a_scale = pyro.param("a_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)

    # Variational parameters for k
    k_loc = pyro.param("k_loc", torch.tensor(0., device=device))
    k_scale = pyro.param("k_scale", torch.tensor(0.5, device=device), constraint=dist.constraints.positive)

    # Variational posterior for sigma_extra
    # if sigma_prior is None:
    #     sigma_m_loc = torch.tensor(0, device=device)
    #     sigma_m_scale = torch.tensor(0, device=device)
    #else:
    sigma_m_loc = pyro.param("sigma_m_loc", torch.tensor(0.5, device=device), constraint=constraints.positive)
    sigma_m_scale = pyro.param("sigma_m_scale", torch.tensor(0.2, device=device), constraint=constraints.positive)

    beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale).to_event(1))
    b0 = pyro.sample("b0", dist.Normal(b0_loc, b0_scale))
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    k = pyro.sample("k", dist.Normal(k_loc, k_scale))
    # if sigma_prior is None:
    #     sigma_m = 0
    #else:
    sigma_m = pyro.sample("sigma_m", dist.LogNormal(torch.log(sigma_m_loc), sigma_m_scale))

    return beta, b0, a, k, sigma_m

class FlambeModel:
    def __init__(self, model, guide, features, hyperparams=None):
        """
        Initialize the FlambeModel with a Pyro model and guide.
        
        Args:
            model (callable): Pyro model function.
            guide (callable): Pyro guide function.
            X_train (torch.Tensor): Training feature matrix.
            y_train (torch.Tensor): Training target values.
            y_err_train (torch.Tensor, optional): Training target errors.
        """
        self.model = model
        self.guide = guide
        self.features = features
        self.N = len(self.features)
        self.param_list = None
        self.params=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hyperparams = hyperparams if hyperparams is not None else {}

        if 'sigma_prior' in self.hyperparams:
            self.sigma_prior = self.hyperparams['sigma_prior']
        else:
            self.sigma_prior = 0.1

    def get_params(self, to_numpy=False, reload=False):
        """
        Get the parameters of the model. Note that if self.params does not yet exist (is None), it looks 
        in the pyro parameter store for the parameters listed in self.param_list. If self.params already exists and reload=False, it simply returns self.params.
        
        Be wary not to accidentally reload parameters from the wrong model!
        
        Returns:
            dict: Dictionary of model parameters.
        """
        
        if not reload and self.params is not None:
            return self.params   
        
        else:         

            if to_numpy:
                # In this case, will put them on the cpu and make them numpy arrays
                # This is useful for plotting and saving the parameters
                self.params = {param: pyro.param(param).detach().clone().cpu().numpy() for param in self.param_list}
            else:
                # In this case, the parameters will remain on the device they were trained on
                # (but detached so they will no longer be updated)
                self.params = {param: pyro.param(param).detach().clone() for param in self.param_list}

            return self.params
    
    def posterior(self):
        
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def sample_posterior(self, N=1):
        
        samples = {}
        
        if N==1:
            for key,distribution in self.posterior().items():
                
                #if key=='sigma_m' and self.sigma_prior is None:
                    #samples[key] = 0
                #else:
                samples[key] = distribution.sample().detach()
                
            return samples
        
        if N > 1:
            for key, distribution in self.posterior().items():
                #if key == 'sigma_m' and self.sigma_prior is None:
                    #samples[key] = np.zeros(N)
                #else:
                samples[key] = distribution.sample((N,)).detach()
                    
            return samples
        
    def train(self, X_train, y_train, y_err_train, num_particles=1, 
              num_iterations=3000, optimizer_type="ClippedAdam", lr=0.02, lrd=0.999,
              return_loss_log=True, mask=None):
        
        """
        Train the FlambeModel using the specified parameters.

        Args:
            num_particles (int): Number of particles for SVI (default=1).
            num_iterations (int): Number of training iterations (default=3000).
            optimizer_type (str): Type of optimizer to use (default="ClippedAdam").
            lr (float): Initial learning rate for the optimizer (default=0.02).
            lrd (float): Learning rate decay for the optimizer (default=0.999).
            
        Returns:
            loss_log (np.ndarray): Array of loss values recorded during training.
        """

        loss_log = train_model(self.model, self.guide, 
                               X_train=X_train, y_train=y_train,
                               y_err_train=y_err_train, 
                               num_particles=num_particles, 
                               num_iterations=num_iterations, 
                               optimizer_type=optimizer_type, 
                               lr=lr, lrd=lrd, mask=mask)
        
        self.get_params()
        
        if return_loss_log:
            return loss_log
        

class LinearModel(FlambeModel):

    def __init__(self, features, hyperparams={}):
        """
        Initialize the LinearModel with the specified features.
        
        Args:
            features (list): List of feature names for the model.
        """

        model = lambda x, y, y_err: linear(x, y, y_err, **hyperparams)
        
        super().__init__(model, linear_guide, features, hyperparams=hyperparams)
        
        self.param_list = ['beta_loc', 'beta_scale', 'b0_loc', 'b0_scale', 'sigma_m_loc', 'sigma_m_scale']
        
    def posterior(self):
        
        '''Defines the posterior distributions of the model parameters.'''
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        else:
            return {'beta': dist.Normal(self.params['beta_loc'], self.params['beta_scale']),
                            'b0': dist.Normal(self.params['b0_loc'], self.params['b0_scale']),
                            'sigma_m': dist.LogNormal(torch.log(self.params['sigma_m_loc']), self.params['sigma_m_scale'])}

    def predict(self, X_test, y_err=None, sigma_m=None, samples=100, summarize=True, ids=None, ci=95):
        
        """
        Predict using the model with sampled parameters.
        
        Args:
            samples (int): Number of samples to draw for predictions (default=100).
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        predictions = []
        for _ in range(samples):
            params = self.sample_posterior()
            beta = params['beta']
            b0 = params['b0']
            if "sigma_m" in params:
                sigma_m = params['sigma_m']
            
            mean = X_test @ beta + b0
            if y_err is None:
                y_err = torch.zeros(1, device=X_test.device)
            total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
            
            pred = pyro.sample("obs", dist.Normal(mean, total_sigma))
            predictions.append(pred.detach().cpu().numpy())
        
        predictions = np.array(predictions)
        
        if summarize:
            
            return summarize_predictions(predictions, ids=ids, ci=ci)
        
        else:
            return predictions

class SigmoidModel(FlambeModel):

    ''' 
    A Bayesian model that assumes a sigmoidal relationship between features and target variable.
    '''
    
    def __init__(self, features, hyperparams={}):
        """
        Initialize SigmoidModel with the specified features.
        
        Args:
            features (list): List of feature names for the model.
        """
   

        model = lambda x, y, y_err: sigmoid(x, y, y_err, **hyperparams)

        guide = lambda x, y, y_err: sigmoid_guide(x, y, y_err)

        super().__init__(model, guide, features, hyperparams=hyperparams)
        
        self.param_list = ['beta_loc', 'beta_scale', 'b0_loc', 'b0_scale', 'a_loc', 'a_scale', 'k_loc', 'k_scale', 'sigma_m_loc', 'sigma_m_scale']
        
    def posterior(self):
        """
        Get the posterior distributions of the model parameters.
        
        Returns:
            dict: Dictionary of posterior distributions for model parameters.
        """
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        else:
            return {
                'beta': dist.Normal(self.params['beta_loc'], self.params['beta_scale']),
                'b0': dist.Normal(self.params['b0_loc'], self.params['b0_scale']),
                'sigma_m': dist.LogNormal(torch.log(self.params['sigma_m_loc']), self.params['sigma_m_scale']) if self.sigma_prior is not None else torch.zeros(1, device-self.device),
                'a': dist.Normal(self.params['a_loc'], self.params['a_scale']),
                'k': dist.Normal(self.params['k_loc'], self.params['k_scale'])
            }
        
    def predict(self, X_test, y_err=None, use_model_error=False, samples=100, summarize=True, ids=None, ci=95):
        
        """
        Predict using the model with sampled parameters.
        
        Args:
            samples (int): Number of samples to draw for predictions (default=100).
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        predictions = []
        for _ in range(samples):
            params = self.sample_posterior()
            beta = params['beta']
            b0 = params['b0']
            if use_model_error:
                sigma_m = params['sigma_m']
            else:
                sigma_m = torch.tensor(0.0, device=X_test.device)
                
            a = params['a']
            k = params['k']
            
            z = X_test @ beta + b0
            mean = a * torch.sigmoid(z.clamp(-10, 10)) + k
            
            if y_err is None:
                y_err = 0
                
            total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
            
            if total_sigma == 0:
                pred = mean
                
            else:
                
                total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
                pred = pyro.sample("obs", dist.Normal(mean, total_sigma))
                
            predictions.append(pred.detach().cpu().numpy())
        
        predictions = np.array(predictions)
        
        if summarize:
            
            return summarize_predictions(predictions, ids=ids, ci=ci)
        
        else:
            return predictions
        
    def y_prior_likelihood(self, distribution='uniform'): 
        
        """
        Given the trained sigmoid and the data that you trained on, generates a prior on the observation space.
        This is useful because due to the nature of the sigmoid many values (including some you observe) are not possible,
        so you can use this prior later to estimate the latent activity (z) values for observed data even if that data is
        outside of the range of the data.
        """
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        
        a = self.params['a']
        k = self.params['k']
        
        if distribution=='uniform':
            return lambda x: dist.Uniform(k, k+a).log_prob(x)
        
    def sample_y_true(self, y_obs, y_err, a=None, k=None, N=1000):
        """
        Sample "true" y values under the model's assumptions given an observed value (y_obs) and its error (y_err).
        
        Args:
            y_obs (float): Observed value.
            y_err (float): Observational error.
            a (float): rescaling parameter from sigmoid transformation
            k (float): intercept parameter from learned sigmoid transformation
            
        """
        
        L = len(y_obs)
        
        if a is None:
            a = self.sample_posterior()['a']
        if k is None:
            k = self.sample_posterior()['k']
        
        min = k
        max = k + a
    
        try:
            truncnorm_dist = truncated_normal(loc=y_obs, scale=y_err, min=min, max=max)
            
            return torch.tensor(truncnorm_dist.rvs(size=(N,L)))
        
        except:
            print("Error in sampling y_true for", y_obs, "with error", y_err)
            
    def transform(self, z):
        
        """
        Sigmoidal transformation function.
        
        Args:
            z (torch.Tensor): Input tensor to transform.
            
        Returns:
            y_pred (torch.Tensor): Transformed tensor.
        """
        a = self.sample_posterior()['a']
        k = self.sample_posterior()['k']
        
        # Apply the sigmoidal transformation
        y_pred = a * torch.sigmoid(z) + k
        
        return y_pred
        
    def inverse_transform(self, y_data, y_err=None, samples=100, summarize=True, ci=95):
        
        """
        Given the overall sigmoidal transformation fit the dataset, sample possible latent activity (z) values for .
        
        Args:
            y_data (torch.Tensor): Input data to transform.
            y_err (torch.Tensor, optional): Error associated with the data (default=None).
            
        Returns:
        """
        
        z_samples = []
        
        for n in range(samples):
            params = self.sample_posterior()
            a = params['a']
            k = params['k']
            
            if y_err is None:
                y_err = torch.zeros(1, device=y_data.device)
                
            # resample data with error (if error passed)
            y_resampled = self.sample_y_true(y_data, y_err, a, k, N=1)[0]
            
            # Compute the inverse transformation
            x = (y_resampled - k) / a
            z_samples.append(torch.log(x / (1 - x)))
            
        z_samples = torch.stack(z_samples, dim=0)  # Shape: (samples, N)
        
        if summarize:
            z_samples = z_samples.detach().cpu().numpy()
            
            # Summarize the samples
            mean_z = np.mean(z_samples, axis=0)
            lower_bound = np.percentile(z_samples, (100 - ci) / 2, axis=0)
            upper_bound = np.percentile(z_samples, 100 - (100 - ci) / 2, axis=0)
            
            return pd.DataFrame({
                'z_mean': mean_z,
                'z_lower_bound': lower_bound,
                'z_upper_bound': upper_bound
            })
            
        else:
            # Return raw samples
            return z_samples.cpu().numpy()
        
# LOOSE MODEL COMPONENTS
        
def gaussian_error_with_halfnormal_prior(y_hat, sigma_prior, y_err=None, error_name="sigma_m"):
    
    '''A simple error model: we merge the observational error (y_err) with a model error (sigma_m) that has a half-normal prior.
    We then assume that the observations are sampled from the mean by a normal distribution with the given error.'''
     
    if sigma_prior is None:
        sigma_m=0
        
    else:
        sigma_m = pyro.sample(error_name, dist.HalfNormal(sigma_prior))
        
    if y_err is None:
        y_err = torch.zeros(1, device=X_train.device)
        
    total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
    
    return dist.Normal(y_hat, total_sigma)



### TRAINING FUNCTION ###

def train_model(model, guide, X_train, y_train, y_err_train, mask=None, num_particles=1, num_iterations=3000, optimizer_type="ClippedAdam", lr=0.02, lrd=0.999):
    
    '''Train a Bayesian model using Pyro with specified parameters. Inputs:
    
        model: Pyro model function
        guide: Pyro guide function
        X_train: Training data features as a tensor (prepared via make_tensors)
        y_train: Training data labels as a tensor (prepared via make_tensors)
        y_err_train: Training data errors as a tensor (prepared via make_tensors)
        
        OPTIONAL:
        num_iterations: Number of training iterations (default=3000). This is the total number of optimization steps.
        num_particles: Number of particles for SVI (default=1). This determines how "jumpy" the gradients are, with higher values leading to more stable but slower convergence.
        optimizer_type: Type of optimizer to use (default="ClippedAdam"). Options are ClippedAdam, Adam, or SGD.
        lr: Initial learning rate for the optimizer (default=0.02).
        lrd: Learning rate decay for the optimizer (default=0.999). This is used to reduce the learning rate over time, which can help with convergence.
            Note that the final learning rate will be lr * lrd^step, where step is the current iteration number. This is only used for ClippedAdam and Adam optimizers.
            
    Returns:
        loss_log: Array of loss values recorded during training.
        Trained parameters are stored in the Pyro parameter store and can be accessed via pyro.get_param_store().
        
    '''

    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the optimizer based on the specified type
    if optimizer_type=="ClippedAdam":
        optimizer = ClippedAdam({
            "lr": lr, 
            "lrd": lrd
        })
        
    elif optimizer_type=="Adam":
        optimizer = Adam({
            "lr": lr, 
            "lrd": lrd
        })
        
    elif optimizer_type=="SGD":
        optimizer = torch.optim.SGD({
            "lr": lr
        })
        
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=num_particles))

    loss_log = []

    for step in tqdm(range(num_iterations)):
        
        if mask is None:
            loss = svi.step(X_train.to(device), y_train.to(device), y_err_train.to(device))
        else:
            loss = svi.step(X_train.to(device), y_train.to(device), y_err_train.to(device), mask=mask.to(device))
        
        loss_log.append(loss)

    print("Training complete!")
    
    return np.array(loss_log)

def two_substrate_sigmoid(X, y=None, y_err=None, mask=None, lambda_w=0.3, a_prior_loc=2.0, a_prior_scale=2.0, k_prior_scale=1.0, sigma_prior=0.1,
                          beta_prior_scale=2, b_prior_scale=2, b_prior_loc=0, beta_s_mask=None):
    """
    Bayesian model for multiple substrates with global and specific effects.
    
    Args:
        X (torch.Tensor): Input feature matrix of shape (N_samples, D)
        y (torch.Tensor): Observed phenotype of shape (N_samples, N) or None
        y_err (torch.Tensor): Experimental standard error of shape (N_samples, N) or None
        mask (torch.Tensor): Binary mask (N_samples, N) where 1 = observed, 0 = missing
        N (int): Number of substrates
        lambda_w (float): Sparsity parameter for specific effects
        a_prior_loc (float): Mean of the prior for `a`
        a_prior_scale (float): Scale of the prior for `a`
        k_prior_scale (float): Scale of the prior for `k`
    
    Returns:
        None (Pyro model for inference)
    """
    device = X.device
    D = X.shape[1]  # Number of features (mutations)
    N=2

    # Priors on global mutation effects (shared across substrates)
    beta_g = pyro.sample("beta_g", dist.Normal(torch.zeros(D, device=device), 
                                               beta_prior_scale * torch.ones(D, device=device)).to_event(1))

    # Priors on substrate-specific mutation effects (sparse)
    beta_s = pyro.sample("beta_s", dist.Laplace(torch.zeros(D, device=device), 
                                                lambda_w * torch.ones(D, device=device)).to_event(1))
    
    if beta_s_mask is not None:
        beta_s = beta_s * beta_s_mask.to(device)
    
    # Priors on offsets, scaling, and shift per substrate
    b = pyro.sample("b", dist.Normal(torch.ones(2, device=device)*b_prior_loc, b_prior_scale*torch.ones(2, device=device)).to_event(1))
    a = pyro.sample("a", dist.Normal(a_prior_loc * torch.ones(N, device=device), 
                                     a_prior_scale * torch.ones(N, device=device)).to_event(1))
    k = pyro.sample("k", dist.Normal(torch.zeros(2, device=device), 
                                     k_prior_scale * torch.ones(2, device=device)).to_event(1))

    # Learnable extra noise term (shared across substrates)
    if sigma_prior is None:
        sigma_m = 0
    else:
        sigma_m = pyro.sample("sigma_m", dist.HalfNormal(torch.ones(2, device=device)*sigma_prior).to_event(1))
        
    #sigma_extra = 0.1

    # Compute global and specific latent variables
    z_g = X @ beta_g
    z_s = X @ beta_s

    # Compute final latent variable per substrate
    z = z_g + torch.cat([torch.zeros(X.shape[0]).unsqueeze(0), z_s.unsqueeze(0)], dim=0) + b.unsqueeze(1)  # Shape: (D, 2)

    # Apply sigmoidal transformation
    mean = a.unsqueeze(1) * torch.sigmoid(z.clamp(-10, 10)) + k.unsqueeze(1)  # Shape: (D, 2)
    mean = mean.T

    # Compute total standard deviation
    y_err = torch.nan_to_num(y_err, nan=0.0)  # Replace NaNs with 0 in y_err
    total_sigma = torch.sqrt(y_err**2 + sigma_m**2)  # Shape: (N_samples, N)
    
    y = torch.where(mask, y, torch.tensor(0.0, device=y.device))  # Replace NaNs with 0 where mask == 0

    # Observed data likelihood (handling missing values)
    with pyro.plate("substrates", 2):  # First plate: substrate dimension (2,)
        with pyro.plate("data", X.shape[0]):  # Second plate: data dimension (26559,)
            pyro.sample("obs", dist.Normal(mean, total_sigma).mask(mask), obs=y)

def two_substrate_sigmoid_guide(X, y=None, y_err=None, mask=None, N=2, lambda_w=0.3, a_prior_loc=2.0, a_prior_scale=2.0, k_prior_scale=2.0,
                                sigma_prior=0.1, beta_prior_scale=2):
    """
    Variational guide for multi-substrate model.
    """
    device = X.device
    D = X.shape[1]

    # Variational parameters for global mutation effects
    beta_g_loc = pyro.param("beta_g_loc", torch.zeros(D, device=device))
    beta_g_scale = pyro.param("beta_g_scale", torch.ones(D, device=device) * 0.5, constraint=constraints.positive)

    # Variational parameters for substrate-specific mutation effects (sparse)
    beta_s_loc = pyro.param("beta_s_loc", torch.zeros(D, device=device))
    beta_s_scale = pyro.param("beta_s_scale", torch.ones(D, device=device) * lambda_w, constraint=constraints.positive)

    # Variational parameters for offsets, scaling, and shift
    b_loc = pyro.param("b_loc", torch.zeros(N, device=device))
    b_scale = pyro.param("b_scale", torch.ones(N, device=device) * 0.5, constraint=constraints.positive)

    a_loc = pyro.param("a_loc", torch.ones(N, device=device), constraint=dist.constraints.positive)
    a_scale = pyro.param("a_scale", torch.ones(N, device=device) * 0.1, constraint=constraints.positive)

    k_loc = pyro.param("k_loc", torch.zeros(N, device=device))
    k_scale = pyro.param("k_scale", torch.ones(N, device=device) * 0.5, constraint=constraints.positive)

    # Variational posterior for sigma_extra
    #sigma_m_loc = pyro.param("sigma_m_loc", torch.ones(N, device=device)*0.1, constraint=constraints.greater_than(1e-6))
    sigma_m_loc = pyro.param("sigma_m_loc", torch.ones(2, device=device)*torch.tensor([0.1, 1.0]), constraint=constraints.positive)
    sigma_m_scale = pyro.param("sigma_m_scale", torch.ones(2, device=device)*0.2, constraint=constraints.positive)

    # Sample parameters
    beta_g = pyro.sample("beta_g", dist.Normal(beta_g_loc, beta_g_scale).to_event(1))
    beta_s = pyro.sample("beta_s", dist.Normal(beta_s_loc, beta_s_scale).to_event(1))
    b = pyro.sample("b", dist.Normal(b_loc, b_scale).to_event(1))
    a = pyro.sample("a", dist.Normal(a_loc, a_scale).to_event(1))
    k = pyro.sample("k", dist.Normal(k_loc, k_scale).to_event(1))
    # Variational posterior for sigma_extra
    #if sigma_prior is None:
        #sigma_m = torch.zeros(2, device=device)
    #else:

    sigma_m = pyro.sample("sigma_m", dist.LogNormal(torch.log(sigma_m_loc), sigma_m_scale).to_event(1))

    return beta_g, beta_s, b, a, k, sigma_m

class TwoSubstrateSpecificityModel(FlambeModel):
    
    '''A model that learns both linear and non-linear effects of mutations on a phenotype using a sigmoid function and pairwise epistasis terms.
    To use this model, you must provide a sparsity mask that indicates which pairwise interactions should be learned.
    By default, it places a Laplace prior on the pairwise interaction terms, which encourages sparsity in the learned interactions.'''
    
    def __init__(self, features, sigma_prior=0.1, beta_prior_scale=2, a_prior_scale=0.5, a_prior_loc=1, k_prior_scale=1,
                 lambda_w=0.3, beta_s_mask=None, b_prior_loc=0, b_prior_scale=2):
        """
        Initialize SigmoidCouplingModel with the specified features.
        
        Args:
            features (list): List of feature names for the model.
        """
            
        model = lambda x, y, y_err, mask: two_substrate_sigmoid(x, y, y_err, mask, beta_prior_scale=beta_prior_scale,
                                                                    a_prior_scale=a_prior_scale,
                                                                    a_prior_loc=a_prior_loc,
                                                                    k_prior_scale=k_prior_scale,
                                                                    b_prior_loc=b_prior_loc,
                                                                    b_prior_scale=b_prior_scale,
                                                                    lambda_w=lambda_w,
                                                                    sigma_prior=sigma_prior,
                                                                    beta_s_mask=beta_s_mask)

        guide = lambda x, y, y_err, mask: two_substrate_sigmoid_guide(x, y, y_err, mask)

        super().__init__(model, guide, features)
        
        self.param_list = ['beta_g_loc', 'beta_g_scale', 'beta_s_loc' ,'beta_s_scale', 'b_loc', 'b_scale', 'a_loc', 'a_scale', 'k_loc', 'k_scale']
        
        if sigma_prior is not None:
            self.param_list += ['sigma_m_loc', 'sigma_m_scale']
            
        # Set model hyperparameters as attributes
        self.sigma_prior = sigma_prior
        self.a_prior_scale = a_prior_scale
        self.a_prior_loc = a_prior_loc
        self.k_prior_scale = k_prior_scale
        self.k_prior_loc = 0
        self.beta_prior_scale = beta_prior_scale
        self.lambda_w = lambda_w
        self.b_prior_loc = b_prior_loc
        self.b_prior_scale = b_prior_scale
        self.beta_s_mask = beta_s_mask

    def posterior(self):
        """
        Get the posterior distributions of the model parameters.
        
        Returns:
            dict: Dictionary of posterior distributions for model parameters.
        """
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        else:
            
            beta_s_loc = self.params['beta_s_loc'] * self.beta_s_mask if self.beta_s_mask is not None else self.params['beta_s_loc']
            
            return {
                "beta_g": dist.Normal(self.params['beta_g_loc'], self.params['beta_g_scale']).to_event(1),
                "beta_s": dist.Normal(beta_s_loc, self.params['beta_s_scale']).to_event(1),
                "b": dist.Normal(self.params['b_loc'], self.params['b_scale']).to_event(1),
                "a": dist.Normal(self.params['a_loc'], self.params['a_scale']).to_event(1),
                "k": dist.Normal(self.params['k_loc'], self.params['k_scale']).to_event(1),
                "sigma_m": dist.LogNormal(torch.log(self.params['sigma_m_loc']), self.params['sigma_m_scale']).to_event(1)
            }
        
    def predict(self, X_test, y_err=None, sigma_m=None, mask=None, samples=100, summarize=True, ids=None, ci=95):
        
        """
        Predict using the model with sampled parameters.
        
        Args:
            samples (int): Number of samples to draw for predictions (default=100).
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        predictions = []
        for _ in range(samples):
            params = self.sample_posterior()
            beta_g = params['beta_g']
            beta_s = params['beta_s']
            
            if self.beta_s_mask is not None:
                beta_s = beta_s * self.beta_s_mask.to(X_test.device)
            
            b = params['b']
            if sigma_m is None:
                sigma_m = torch.zeros(params['sigma_m'].shape, device=X_test.device)
            else:
                sigma_m = params['sigma_m']
            a = params['a']
            k = params['k']
            
            z_g = X_test @ beta_g
            z_s = X_test @ beta_s
            z = z = z_g + torch.cat([torch.zeros(X_test.shape[0]).unsqueeze(0), z_s.unsqueeze(0)], dim=0) + b.unsqueeze(1)
            mean = a.unsqueeze(1) * torch.sigmoid(z.clamp(-10, 10)) + k.unsqueeze(1)
            
            if y_err is None:
                y_err = torch.zeros(X_test.shape[0], device=X_test.device)

            total_sigma = torch.sqrt(y_err**2 + sigma_m.unsqueeze(1)**2)
            
            #print(total_sigma.shape, mean.shape)
            
            if np.any(total_sigma.numpy()==0):
                total_sigma += 1e-6
            pred = pyro.sample("obs", dist.Normal(mean, total_sigma))
            predictions.append(pred.detach().cpu().numpy())
        
        predictions = np.array(predictions)
        
        if summarize:
            
            pred1 = summarize_predictions(predictions[:,0], label='1', ids=ids, ci=ci)
            pred2 = summarize_predictions(predictions[:,1], label='2', ids=ids, ci=ci)
            return pd.concat([pred1, pred2], axis=1)

        else:
            return predictions
        
    def sample_y_true(model, y_obs, y_err, a=None, k=None, N=1000, n=0):
        """
        Sample "true" y values under the model's assumptions given an observed value (y_obs) and its error (y_err).
        
        Args:
            y_obs (float): Observed value.
            y_err (float): Observational error.
            a (float): rescaling parameter from sigmoid transformation
            k (float): intercept parameter from learned sigmoid transformation
            
        """
        
        L = len(y_obs)
        
        if a is None:
            a = model.sample_posterior()['a'][n]
        if k is None:
            k = model.sample_posterior()['k'][n]
        
        min = k
        max = k + a

        truncnorm_dist = utils.truncated_normal(loc=y_obs, scale=y_err, min=min, max=max)
        
        return torch.tensor(truncnorm_dist.rvs(size=(N,L)))
    
    def sample_y_true(self, y_obs, y_err, a=None, k=None, N=1000, n=0):
    
        """
        Sample "true" y values under the model's assumptions given an observed value (y_obs) and its error (y_err).
        
        Args:
            y_obs (float): Observed value.
            y_err (float): Observational error.
            a (float): rescaling parameter from sigmoid transformation
            k (float): intercept parameter from learned sigmoid transformation
            
        """
        
        L = len(y_obs)
        
        if a is None:
            a = self.sample_posterior()['a'][n]
        if k is None:
            k = self.sample_posterior()['k'][n]
        
        min = k
        max = k + a

        truncnorm_dist = truncated_normal(loc=y_obs, scale=y_err, min=min, max=max)
        
        return torch.tensor(truncnorm_dist.rvs(size=(N,L)))
    
    def inverse_transform(self, y_data, y_err=None, samples=100, summarize=True, ci=95, n_subs=0):
        
        """
        Given the overall sigmoidal transformation fit the dataset, sample possible latent activity (z) values for .
        
        Args:
            y_data (torch.Tensor): Input data to transform.
            y_err (torch.Tensor, optional): Error associated with the data (default=None).
            
        Returns:
        """
        
        z_samples = []
        
        for n in range(samples):
            params = self.sample_posterior()
            a = params['a'][n_subs]
            k = params['k'][n_subs]
            #b = params['b'][n_subs]

            if y_err is None:
                y_err = torch.zeros(1, device=y_data.device)
                
            # resample data with error (if error passed)
            y_resampled = self.sample_y_true(y_data, y_err, a, k, N=1)[0]
            
            # Compute the inverse transformation
            x = (y_resampled - k) / a
            z_samples.append(torch.log(x / (1 - x)))

        z_samples = torch.stack(z_samples, dim=0)  # Shape: (samples, N)
        
        if summarize:
            z_samples = z_samples.detach().cpu().numpy()
            
            # Compute the mean and credible intervals
            mean_z = np.mean(z_samples, axis=0)
            lower_bound = np.percentile(z_samples, (100 - ci) / 2, axis=0)
            upper_bound = np.percentile(z_samples, 100 - (100 - ci) / 2, axis=0)
            
            return pd.DataFrame({
                'z_mean': mean_z,
                'z_lower_bound': lower_bound,
                'z_upper_bound': upper_bound
            })
            
        else:
            # Return raw samples
            return z_samples
        
    def transform(self, z_g, z_s=None):

        """
        Transform latent variables (z_g, z_s) back to the original space.
        """
        a = self.sample_posterior()['a']
        k = self.sample_posterior()['k']
        b = self.sample_posterior()['b']
        
        if z_s is None:
            z_s = torch.zeros(z_g.shape, device=z_g.device)

        z = z_g + torch.cat([torch.zeros(z_g.shape[0]).unsqueeze(0), z_s.unsqueeze(0)], dim=0) + b.unsqueeze(1)

        # Apply the inverse of the sigmoidal transformation
        y_transformed = a * torch.sigmoid(z.T) + k

        return y_transformed

def multi_substrate_sparse_sigmoid(X, y=None, y_err=None, mask=None, N=2, lambda_w=0.3, a_prior_shape=2.0, a_prior_scale=4.0, k_prior_scale=1.0, sigma_prior=0.01,
                                   beta_prior_scale=2.):
    """
    Bayesian model for multiple substrates with global and specific effects.
    
    Args:
        X (torch.Tensor): Input feature matrix of shape (N_samples, D)
        y (torch.Tensor): Observed phenotype of shape (N_samples, N) or None
        y_err (torch.Tensor): Experimental standard error of shape (N_samples, N) or None
        mask (torch.Tensor): Binary mask (N_samples, N) where 1 = observed, 0 = missing
        N (int): Number of substrates
        lambda_w (float): Sparsity parameter for specific effects
        a_prior_loc (float): Mean of the prior for `a`
        a_prior_scale (float): Scale of the prior for `a`
        k_prior_scale (float): Scale of the prior for `k`
    
    Returns:
        None (Pyro model for inference)
    """
    device = X.device
    D = X.shape[1]  # Number of features (mutations)
    N = y.shape[1] if y is not None else N
    
    # Priors on global mutation effects (shared across substrates)
    beta_g = pyro.sample("beta_g", dist.Normal(torch.zeros(D, device=device), 
                                               beta_prior_scale * torch.ones(D, device=device)).to_event(1))

    # Priors on substrate-specific mutation effects (sparse)
    beta_s = pyro.sample("beta_s", dist.Laplace(torch.zeros(D, N, device=device), 
                                                lambda_w * torch.ones(D, N, device=device)).to_event(2))

    # Offset per substrate - single scaling and shift term
    b = pyro.sample("b", dist.Normal(torch.zeros(N, device=device), torch.ones(N, device=device)).to_event(1))
    
    a = pyro.sample("a", dist.InverseGamma(a_prior_shape * torch.ones(1, device=device), 
                                     a_prior_scale * torch.ones(1, device=device)))
    k = pyro.sample("k", dist.Normal(torch.tensor(0.0, device=device),
                                     torch.tensor(k_prior_scale, device=device)))

    # Learnable extra noise term (shared across substrates)
    if sigma_prior is None:
        sigma_m = 0
    else:
        sigma_m = pyro.sample("sigma_m", dist.HalfNormal(torch.tensor(sigma_prior, device=device)))

    # Compute global and specific latent variables
    z_g = X @ beta_g  # Shape: (N_samples,)
    z_s = X @ beta_s  # Shape: (N_samples, N)

    # Compute final latent variable per substrate
    z = z_g[:, None] + z_s + b  # Shape: (N_samples, N)

    # Apply sigmoidal transformation
    mean = a * torch.sigmoid(z.clamp(-10, 10)) + k  # Shape: (N_samples, N)

    # Compute total standard deviation
    y_err = torch.nan_to_num(y_err, nan=0.0)  # Replace NaNs with 0 in y_err
    total_sigma = torch.sqrt(y_err**2 + sigma_m**2)  # Shape: (N_samples, N)
    
    y = torch.where(mask, y, torch.tensor(0.0, device=y.device))  # Replace NaNs with 0 where mask == 0

    # Observed data likelihood (handling missing values)
    with pyro.plate("substrates", N):  # First plate: substrate dimension (2,)
        with pyro.plate("data", X.shape[0]):  # Second plate: data dimension (26559,)
            pyro.sample("obs", dist.Normal(mean, total_sigma).mask(mask), obs=y)

def multi_substrate_sigmoid_guide(X, y=None, y_err=None, mask=None, N=2, lambda_w=0.3, a_prior_shape=2.0, a_prior_scale=1.0, k_prior_scale=2.0, sigma_prior=0.01,
                                  beta_prior_scale=2):
    """
    Variational guide for multi-substrate model.
    """
    device = X.device
    D = X.shape[1]
    N = y.shape[1]

    # Variational parameters for global mutation effects
    beta_g_loc = pyro.param("beta_g_loc", torch.zeros(D, device=device))
    beta_g_scale = pyro.param("beta_g_scale", torch.ones(D, device=device) * 0.5, constraint=constraints.positive)

    # Variational parameters for substrate-specific mutation effects (sparse)
    beta_s_loc = pyro.param("beta_s_loc", torch.zeros(D, N, device=device))
    beta_s_scale = pyro.param("beta_s_scale", torch.ones(D, N, device=device) * lambda_w, constraint=constraints.positive)

    # Variational parameters for offsets, scaling, and shift
    b_loc = pyro.param("b_loc", torch.zeros(N, device=device))
    b_scale = pyro.param("b_scale", torch.ones(N, device=device) * 0.5, constraint=constraints.positive)

    a_loc = pyro.param("a_loc", torch.tensor(1., device=device), constraint=dist.constraints.positive)
    a_scale = pyro.param("a_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)

    k_loc = pyro.param("k_loc", torch.tensor(0., device=device))
    k_scale = pyro.param("k_scale", torch.tensor(0.5, device=device), constraint=constraints.positive)

    # Variational posterior for sigma_extra - FIXED: Remove .to_event(1) for scalar
    sigma_m_loc = pyro.param("sigma_m_loc", torch.tensor(0.01, device=device), constraint=constraints.positive)
    sigma_m_scale = pyro.param("sigma_m_scale", torch.tensor(0.02, device=device), constraint=constraints.positive)

    # Sample parameters
    beta_g = pyro.sample("beta_g", dist.Normal(beta_g_loc, beta_g_scale).to_event(1))
    beta_s = pyro.sample("beta_s", dist.Normal(beta_s_loc, beta_s_scale).to_event(2))
    b = pyro.sample("b", dist.Normal(b_loc, b_scale).to_event(1))
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    k = pyro.sample("k", dist.Normal(k_loc, k_scale))
    
    # FIXED: Remove .to_event(1) since sigma_m is scalar
    sigma_m = pyro.sample("sigma_m", dist.LogNormal(torch.log(sigma_m_loc), sigma_m_scale))
    
    return beta_g, beta_s, b, a, k, sigma_m

class MultiSubstrateSpecificityModel(FlambeModel):
    
    def __init__(self, features, sigma_prior=0.1, beta_prior_scale=2, a_prior_scale=0.5, a_prior_shape=1, k_prior_scale=1,
                 lambda_w=0.3, mask=None):
        """
        Initialize SigmoidCouplingModel with the specified features.
        
        Args:
            features (list): List of feature names for the model.
        """
            
        model = lambda x, y, y_err, mask: multi_substrate_sparse_sigmoid(x, y, y_err, mask, beta_prior_scale=beta_prior_scale,
                                                                    a_prior_scale=a_prior_scale,
                                                                    a_prior_shape=a_prior_shape,
                                                                    k_prior_scale=k_prior_scale)

        guide = lambda x, y, y_err, mask: multi_substrate_sigmoid_guide(x, y, y_err, mask)

            
        super().__init__(model, guide, features)
        
        self.param_list = ['beta_g_loc', 'beta_g_scale', 'beta_s_loc' ,'beta_s_scale', 'b_loc', 'b_scale', 'a_loc', 'a_scale', 'k_loc', 'k_scale']
        
        if sigma_prior is not None:
            self.param_list += ['sigma_m_loc', 'sigma_m_scale']
            
    
    def posterior(self):
        """
        Get the posterior distributions of the model parameters.
        
        Returns:
            dict: Dictionary of posterior distributions for model parameters.
        """
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        else:
            return {
                "beta_g": dist.Normal(pyro.param("beta_g_loc"), pyro.param("beta_g_scale")).to_event(1),
                "beta_s": dist.Normal(pyro.param("beta_s_loc"), pyro.param("beta_s_scale")).to_event(2),
                "b": dist.Normal(pyro.param("b_loc"), pyro.param("b_scale")).to_event(1),
                "a": dist.Normal(pyro.param("a_loc"), pyro.param("a_scale")).to_event(1),
                "k": dist.Normal(pyro.param("k_loc"), pyro.param("k_scale")).to_event(1),
                "sigma_m": dist.LogNormal(torch.log(pyro.param("sigma_m_loc")), pyro.param("sigma_m_scale"))
            }
        
    def predict(self, X_test, y_err=None, sigma_m=None, mask=None, samples=100, summarize=True, ids=None, ci=95):
        
        """
        Predict using the model with sampled parameters.
        
        Args:
            samples (int): Number of samples to draw for predictions (default=100).
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        predictions = []
        for _ in range(samples):
            params = self.sample_posterior()
            beta_g = params['beta_g']
            beta_s = params['beta_s']
            b = params['b']
            if sigma_m is not None:
                sigma_m = params['sigma_m']
            a = params['a']
            k = params['k']
            
            z_g = X_test @ beta_g
            z_s = X_test @ beta_s
            z = z_g + torch.cat([torch.zeros(X_test.shape[1]).unsqueeze(0), z_s.unsqueeze(0)], dim=0) + b
            mean = a * torch.sigmoid(z.clamp(-10, 10)) + k
            
            if y_err is None:
                y_err = 0
            total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
            
            pred = pyro.sample("obs", dist.Normal(mean, total_sigma))
            predictions.append(pred.detach().cpu().numpy())
        
        predictions = np.array(predictions)
        
        if summarize:
            
            return summarize_predictions(predictions, ids=ids, ci=ci)
        
        else:
            return predictions
        
        
        

def two_substrate_competitive_sigmoid(X, y=None, y_err=None, mask=None, lambda_w=0.3, a_prior_loc=2.0, a_prior_scale=2.0, k_prior_scale=1.0, sigma_prior=0.1,
                          beta_prior_scale=2, b_prior_scale=2, b_prior_loc=0):
    """
    Bayesian model for multiple substrates with global and specific effects.
    
    Args:
        X (torch.Tensor): Input feature matrix of shape (N_samples, D)
        y (torch.Tensor): Observed phenotype of shape (N_samples, N) or None
        y_err (torch.Tensor): Experimental standard error of shape (N_samples, N) or None
        mask (torch.Tensor): Binary mask (N_samples, N) where 1 = observed, 0 = missing
        N (int): Number of substrates
        lambda_w (float): Sparsity parameter for specific effects
        a_prior_loc (float): Mean of the prior for `a`
        a_prior_scale (float): Scale of the prior for `a`
        k_prior_scale (float): Scale of the prior for `k`
    
    Returns:
        None (Pyro model for inference)
    """
    device = X.device
    D = X.shape[1]  # Number of features (mutations)
    N=2

    # Priors on global mutation effects (shared across substrates)
    beta_g = pyro.sample("beta_g", dist.Normal(torch.zeros(D, device=device), 
                                               beta_prior_scale * torch.ones(D, device=device)).to_event(1))

    # Priors on substrate-specific mutation effects (sparse)
    beta_s = pyro.sample("beta_s", dist.Laplace(torch.zeros(D, device=device), 
                                                lambda_w * torch.ones(D, device=device)).to_event(1))
    
    # Priors on offsets, scaling, and shift per substrate
    b = pyro.sample("b", dist.Normal(torch.ones(2, device=device)*b_prior_loc, b_prior_scale*torch.ones(2, device=device)).to_event(1))
    a = pyro.sample("a", dist.Normal(a_prior_loc * torch.ones(N, device=device), 
                                     a_prior_scale * torch.ones(N, device=device)).to_event(1))
    k = pyro.sample("k", dist.Normal(torch.zeros(2, device=device), 
                                     k_prior_scale * torch.ones(2, device=device)).to_event(1))
    
    I = pyro.sample("I", dist.Normal(torch.zeros(1, device=device), 0.2*torch.ones(1, device=device)))

    # Learnable extra noise term (shared across substrates)
    if sigma_prior is None:
        sigma_m = 0
    else:
        sigma_m = pyro.sample("sigma_m", dist.HalfNormal(torch.ones(2, device=device)*sigma_prior).to_event(1))
        

    #sigma_extra = 0.1

    # Compute global and specific latent variables
    z_g = X @ beta_g
    z_s = X @ beta_s

    # Compute final latent variable per substrate
    z = z_g + torch.cat([torch.zeros(X.shape[0]).unsqueeze(0), z_s.unsqueeze(0)], dim=0) + b.unsqueeze(1)  # Shape: (D, 2)

    # Apply sigmoidal transformation
    mean = a.unsqueeze(1) * torch.sigmoid(z.clamp(-10, 10)) + k.unsqueeze(1)  # Shape: (D, 2)
    mean = mean.T

    adjusted_mean = torch.cat([(mean[:, 0]*(1-I*mean[:,1])).unsqueeze(1), mean[:, 1].unsqueeze(1)], dim=1)

    # Compute total standard deviation
    y_err = torch.nan_to_num(y_err, nan=0.0)  # Replace NaNs with 0 in y_err
    total_sigma = torch.sqrt(y_err**2 + sigma_m**2)  # Shape: (N_samples, N)
    
    y = torch.where(mask, y, torch.tensor(0.0, device=y.device))  # Replace NaNs with 0 where mask == 0

    # Observed data likelihood (handling missing values)
    with pyro.plate("substrates", 2):  # First plate: substrate dimension (2,)
        with pyro.plate("data", X.shape[0]):  # Second plate: data dimension (26559,)
            pyro.sample("obs", dist.Normal(adjusted_mean, total_sigma).mask(mask), obs=y)

def two_substrate_competitive_sigmoid_guide(X, y=None, y_err=None, mask=None, N=2, lambda_w=0.3, a_prior_loc=2.0, a_prior_scale=2.0, k_prior_scale=2.0,
                                sigma_prior=0.1, beta_prior_scale=2):
    """
    Variational guide for multi-substrate model.
    """
    device = X.device
    D = X.shape[1]

    # Variational parameters for global mutation effects
    beta_g_loc = pyro.param("beta_g_loc", torch.zeros(D, device=device))
    beta_g_scale = pyro.param("beta_g_scale", torch.ones(D, device=device) * 0.5, constraint=constraints.positive)

    # Variational parameters for substrate-specific mutation effects (sparse)
    beta_s_loc = pyro.param("beta_s_loc", torch.zeros(D, device=device))
    beta_s_scale = pyro.param("beta_s_scale", torch.ones(D, device=device) * lambda_w, constraint=constraints.positive)

    # Variational parameters for offsets, scaling, and shift
    b_loc = pyro.param("b_loc", torch.zeros(N, device=device))
    b_scale = pyro.param("b_scale", torch.ones(N, device=device) * 0.5, constraint=constraints.positive)

    a_loc = pyro.param("a_loc", torch.ones(N, device=device), constraint=dist.constraints.positive)
    a_scale = pyro.param("a_scale", torch.ones(N, device=device) * 0.1, constraint=constraints.positive)

    k_loc = pyro.param("k_loc", torch.zeros(N, device=device))
    k_scale = pyro.param("k_scale", torch.ones(N, device=device) * 0.5, constraint=constraints.positive)
    
    I_loc = pyro.param("I_loc", torch.zeros(1, device=device))
    I_scale = pyro.param("I_scale", torch.ones(1, device=device) * 0.3, constraint=constraints.positive)

    # Variational posterior for sigma_extra
    #sigma_m_loc = pyro.param("sigma_m_loc", torch.ones(N, device=device)*0.1, constraint=constraints.greater_than(1e-6))
    sigma_m_loc = pyro.param("sigma_m_loc", torch.ones(2, device=device)*torch.tensor([0.1, 1.0]), constraint=constraints.positive)
    sigma_m_scale = pyro.param("sigma_m_scale", torch.ones(2, device=device)*0.2, constraint=constraints.positive)

    # Sample parameters
    beta_g = pyro.sample("beta_g", dist.Normal(beta_g_loc, beta_g_scale).to_event(1))
    beta_s = pyro.sample("beta_s", dist.Normal(beta_s_loc, beta_s_scale).to_event(1))
    b = pyro.sample("b", dist.Normal(b_loc, b_scale).to_event(1))
    a = pyro.sample("a", dist.Normal(a_loc, a_scale).to_event(1))
    k = pyro.sample("k", dist.Normal(k_loc, k_scale).to_event(1))
    I = pyro.sample("I", dist.Normal(I_loc, I_scale))

    sigma_m = pyro.sample("sigma_m", dist.LogNormal(torch.log(sigma_m_loc), sigma_m_scale).to_event(1))

    return beta_g, beta_s, b, a, k, I, sigma_m


class TwoSubstrateCompetitiveSpecificityModel(FlambeModel):
    
    '''A model that learns both linear and non-linear effects of mutations on a phenotype using a sigmoid function and pairwise epistasis terms.
    To use this model, you must provide a sparsity mask that indicates which pairwise interactions should be learned.
    By default, it places a Laplace prior on the pairwise interaction terms, which encourages sparsity in the learned interactions.'''
    
    def __init__(self, features, sigma_prior=0.1, beta_prior_scale=2, a_prior_scale=0.5, a_prior_loc=1, k_prior_scale=1,
                 lambda_w=0.3, mask=None, b_prior_loc=0, b_prior_scale=2):
        """
        Initialize SigmoidCouplingModel with the specified features.
        
        Args:
            features (list): List of feature names for the model.
        """
            
        model = lambda x, y, y_err, mask: two_substrate_competitive_sigmoid(x, y, y_err, mask, beta_prior_scale=beta_prior_scale,
                                                                    a_prior_scale=a_prior_scale,
                                                                    a_prior_loc=a_prior_loc,
                                                                    k_prior_scale=k_prior_scale,
                                                                    b_prior_loc=b_prior_loc,
                                                                    b_prior_scale=b_prior_scale,
                                                                    lambda_w=lambda_w,
                                                                    sigma_prior=sigma_prior)
    
        guide = lambda x, y, y_err, mask: two_substrate_competitive_sigmoid_guide(x, y, y_err, mask)

            
        super().__init__(model, guide, features)
        
        self.param_list = ['beta_g_loc', 'beta_g_scale', 'beta_s_loc' ,'beta_s_scale', 'b_loc', 'b_scale', 'a_loc', 'a_scale', 'k_loc', 'k_scale', "I_loc", "I_scale"]
        
        if sigma_prior is not None:
            self.param_list += ['sigma_m_loc', 'sigma_m_scale']
            
    def posterior(self):
        """
        Get the posterior distributions of the model parameters.
        
        Returns:
            dict: Dictionary of posterior distributions for model parameters.
        """
        
        if self.params is None:
            raise ValueError("Model parameters have not been trained yet. Fit some data first!")
        else:
            return {
                "beta_g": dist.Normal(pyro.param("beta_g_loc"), pyro.param("beta_g_scale")).to_event(1),
                "beta_s": dist.Normal(pyro.param("beta_s_loc"), pyro.param("beta_s_scale")).to_event(1),
                "b": dist.Normal(pyro.param("b_loc"), pyro.param("b_scale")).to_event(1),
                "a": dist.Normal(pyro.param("a_loc"), pyro.param("a_scale")).to_event(1),
                "k": dist.Normal(pyro.param("k_loc"), pyro.param("k_scale")).to_event(1),
                "sigma_m": dist.LogNormal(torch.log(pyro.param("sigma_m_loc")), pyro.param("sigma_m_scale")).to_event(1),
                "I": dist.Normal(pyro.param("I_loc"), pyro.param("I_scale")).to_event(1)
            }
        
    def predict(self, X_test, y_err=None, sigma_m=None, mask=None, samples=100, summarize=True, ids=None, ci=95):
        
        """
        Predict using the model with sampled parameters.
        
        Args:
            samples (int): Number of samples to draw for predictions (default=100).
            
        Returns:
            np.ndarray: Array of predicted values.
        """
        predictions = []
        for _ in range(samples):
            params = self.sample_posterior()
            beta_g = params['beta_g']
            beta_s = params['beta_s']
            b = params['b']
            if sigma_m is not None:
                sigma_m = params['sigma_m']
            a = params['a']
            k = params['k']
            I = params['I']
            
            z_g = X_test @ beta_g
            z_s = X_test @ beta_s
            z = z = z_g + torch.cat([torch.zeros(X.shape[0]).unsqueeze(0), z_s.unsqueeze(0)], dim=0)
            mean = a.unsqueeze(1) * torch.sigmoid(z.clamp(-10, 10)) + k.unsqueeze(1)
            mean[0] = mean[0] * (1-mean[1]*I)
            
            if y_err is None:
                y_err = 0
            total_sigma = torch.sqrt(y_err**2 + sigma_m**2)
            
            pred = pyro.sample("obs", dist.Normal(mean, total_sigma))
            predictions.append(pred.detach().cpu().numpy())
        
        predictions = np.array(predictions)
        
        if summarize:
            
            return summarize_predictions(predictions, ids=ids, ci=ci)
        
        else:
            return predictions
        
    def sample_y_true(model, y_obs, y_err, a=None, k=None, N=1000, n=0):
        """
        Sample "true" y values under the model's assumptions given an observed value (y_obs) and its error (y_err).
        
        Args:
            y_obs (float): Observed value.
            y_err (float): Observational error.
            a (float): rescaling parameter from sigmoid transformation
            k (float): intercept parameter from learned sigmoid transformation
            
        """
        
        L = len(y_obs)
        
        if a is None:
            a = model.sample_posterior()['a'][n]
        if k is None:
            k = model.sample_posterior()['k'][n]
        
        min = k
        max = k + a

        truncnorm_dist = utils.truncated_normal(loc=y_obs, scale=y_err, min=min, max=max)
        
        return torch.tensor(truncnorm_dist.rvs(size=(N,L)))