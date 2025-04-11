# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file implements the Bayesian Flow and BFN loss for continuous and discrete variables.
Finally it implements the BFN using these objects.
For consistency we use always use a tuple to store input parameters.
It has just one element for discrete data (the probabilities) and two for continuous/discretized (mean & variance).
The probability distributions and network architectures are defined in probability.py and networks dir.
"Cts" is an abbreviation of "Continuous".
"""

import math
from abc import abstractmethod, ABC
from typing import Union, Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from probability import (
    DiscreteDistributionFactory,
    CtsDistributionFactory,
    PredDistToDataDistFactory,
    DiscretizedCtsDistribution,
)
from utils_model import sandwich, float_to_idx

class BayesianFlow(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor, ...]:
        """Returns the initial input params (for a batch) at t=0. Used during sampling.
        For discrete data, the tuple has length 1 and contains the initial class probabilities.
        For continuous data, the tuple has length 2 and contains the mean and precision.
        
        """
        pass

    @abstractmethod
    def params_to_net_inputs(self, params: tuple[Tensor, ...]) -> Tensor:
        """Utility method to convert input distribution params to network inputs if needed.
        """
        pass

    @abstractmethod
    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> float:
        """Returns the alpha at step i of total n_steps according to the flow schedule. Used:
        a) during sampling, when i and alpha are the same for all samples in the batch.
        b) during discrete time loss computation, when i and alpha are different for samples in the batch.
        
        """
        pass

    @abstractmethod
    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        """Returns the sender distribution with accuracy alpha obtained by adding appropriate noise to the data x. Used:
        a) during sampling (same alpha for whole batch) to sample from the output distribution produced by the net.
        b) during discrete time loss computation when alpha are different for samples in the batch.
        
        """
        pass
    
    @abstractmethod
    def get_sender_dist_copy(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        """Returns the sender distribution with accuracy alpha obtained by adding appropriate noise to the data x. Used:
        a) during sampling (same alpha for whole batch) to sample from the output distribution produced by the net.
        b) during discrete time loss computation when alpha are different for samples in the batch.
        
        """
        pass

    @abstractmethod
    def update_input_params(self, input_params: tuple[Tensor, ...], y: Tensor, alpha: float) -> tuple[Tensor, ...]:
        """Updates the distribution parameters using Bayes' theorem in light of noisy sample y.
        Used during sampling when alpha is the same for the whole batch.
        
        """        
        pass

    @abstractmethod
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        """Returns a sample from the Bayesian Flow distribution over input parameters at time t conditioned on data.
        Used during training when t (and thus accuracies) are different for different samples in the batch.
        For discrete data, the returned tuple has length 1 and contains the class probabilities.
        For continuous data, the returned tuple has length 2 and contains the mean and precision.
        
        """
        pass


class Loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor) -> Tensor:
        """Returns the continuous time KL loss (and any other losses) at time t (between 0 and 1).
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass

    @abstractmethod
    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples: int = 20
    ) -> Tensor:
        """Returns the discrete time KL loss for n_steps total of communication at time t (between 0 and 1) using
        n_samples for Monte Carlo estimation of the discrete loss.
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass

    @abstractmethod
    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        """Returns the reconstruction loss, i.e. the final cost of transmitting clean data.
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass

# Discrete Data


class DiscreteBayesianFlow(BayesianFlow):
    def __init__(
        self,
        n_classes: int,
        min_sqrt_beta: float = 1e-10,
        discretize: bool = False,
        epsilon: float = 1e-6,
        max_sqrt_beta: float = 1,
    ):
        super().__init__()
        self.n_classes = n_classes  # K
        self.epsilon = epsilon
        self.discretize = discretize
        self.min_sqrt_beta = min_sqrt_beta
        self.max_sqrt_beta = max_sqrt_beta
        self.uniform_entropy = math.log(self.n_classes)

    def t_to_sqrt_beta(self, t):
        """sqrt{\beta(t)} = t \sqrt{\beta(1)}."""
           
        return t * self.max_sqrt_beta

    def count_dist(self, x, beta=None):        
        mean = (self.n_classes * F.one_hot(x.long(), self.n_classes)) - 1
        std_dev = math.sqrt(self.n_classes)
        if beta is not None:
            mean = mean * beta
            std_dev = std_dev * beta.sqrt()
        return D.Normal(mean, std_dev, validate_args=False)

    def count_sample(self, x, beta):     
        return self.count_dist(x, beta).rsample(), self.count_dist(x, beta)

    @torch.no_grad()
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor]:
        return (torch.ones(*data_shape, self.n_classes, device=device) / self.n_classes,)

    @torch.no_grad()
    def params_to_net_inputs(self, params: tuple[Tensor]) -> Tensor:
        params = params[0]
        if self.n_classes == 2:
            params = params * 2 - 1  # We scale-shift here for MNIST instead of in the network like for text
            params = params[..., :1]
        return params

    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> Union[float, Tensor]:
        return ((self.max_sqrt_beta / n_steps) ** 2) * (2 * i - 1)

    def get_sender_dist(self, e_x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        # e_x = F.one_hot(x.long(), self.n_classes)
        alpha = alpha.unsqueeze(-1) if isinstance(alpha, Tensor) else alpha
        dist = D.Normal(alpha * ((self.n_classes * e_x) - 1), (self.n_classes * alpha) ** 0.5)
        return dist
    
    def get_sender_dist_copy(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        e_x = F.one_hot(x.long(), self.n_classes)
        alpha = alpha.unsqueeze(-1) if isinstance(alpha, Tensor) else alpha
        dist = D.Normal(alpha * ((self.n_classes * e_x) - 1), (self.n_classes * alpha) ** 0.5)
        return dist

    def update_input_params(self, input_params: tuple[Tensor], y: Tensor, alpha: float) -> tuple[Tensor]:
        new_input_params = input_params[0] * y.exp()
        new_input_params /= new_input_params.sum(-1, keepdims=True)
        return (new_input_params,)

    @torch.no_grad()
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor]:
        
        if self.discretize:
            data = float_to_idx(data, self.n_classes)
        
        # \sqrt{\beta(t)}
        sqrt_beta = self.t_to_sqrt_beta(t.clamp(max=1 - self.epsilon))
        lo_beta = sqrt_beta < self.min_sqrt_beta
        sqrt_beta = sqrt_beta.clamp(min=self.min_sqrt_beta)
        beta = sqrt_beta.square().unsqueeze(-1)
        
        logits = self.count_sample(data, beta)
        probs = F.softmax(logits, -1)
        probs = torch.where(lo_beta.unsqueeze(-1), torch.ones_like(probs) / self.n_classes, probs)
        
        if self.n_classes == 2:
            probs = probs[..., :1]
            probs = probs.reshape_as(data)
        input_params = (probs,)
        return input_params


class DiscreteBayesianFlowLoss(Loss):
    def __init__(
        self,
        bayesian_flow: DiscreteBayesianFlow,
        distribution_factory: DiscreteDistributionFactory,
    ):
        super().__init__()
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        self.K = self.bayesian_flow.n_classes

    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t) -> Tensor:
        flat_output = sandwich(output_params)
        pred_probs = self.distribution_factory.get_dist(flat_output).probs
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        tgt_mean = torch.nn.functional.one_hot(flat_target.long(), self.K)
        kl = self.K * ((tgt_mean - pred_probs).square()).sum(-1)
        t = t.flatten(start_dim=1).float()
        loss = t * (self.bayesian_flow.max_sqrt_beta**2) * kl
        return loss

    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        i = t * n_steps + 1
        alpha = self.bayesian_flow.get_alpha(i, n_steps).flatten(start_dim=1)
        sender_dist = self.bayesian_flow.get_sender_dist(flat_target, alpha)  # sender dist.

        flat_output = sandwich(output_params)
        receiver_mix_wts = self.distribution_factory.get_dist(flat_output).probs
        receiver_mix_dist = D.Categorical(probs=receiver_mix_wts.unsqueeze(-2))
        classes = torch.arange(self.K, device=flat_target.device).long().unsqueeze(0).unsqueeze(0)
        receiver_components = self.bayesian_flow.get_sender_dist(classes, alpha.unsqueeze(-1))
        receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components)

        y = sender_dist.sample(torch.Size([n_samples]))
        loss = n_steps * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).sum(-1).mean(1, keepdims=True)
        return loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        flat_outputs = sandwich(output_params)
        flat_data = data.flatten(start_dim=1)
        output_dist = self.distribution_factory.get_dist(flat_outputs)
        return -output_dist.log_prob(flat_data)


class BFN(nn.Module):
    def __init__(self, net: nn.Module, bayesian_flow: BayesianFlow, loss: Loss):
        super().__init__()
        
        self.net = net
        self.bayesian_flow = bayesian_flow
        self.loss = loss
        
        self.num_classes = bayesian_flow.n_classes
        
        self.mse_loss = nn.MSELoss()
        

    @staticmethod
    @torch.no_grad()
    def sample_t(data: Tensor, n_steps: Optional[int]) -> Tensor:
        
        if n_steps == 0 or n_steps is None:
            t = torch.rand(data.size(0), device=data.device).unsqueeze(-1)
        else:
            t = torch.randint(0, n_steps, (data.size(0),), device=data.device).unsqueeze(-1) / n_steps 
        t = (torch.ones_like(data).flatten(start_dim=1) * t).reshape_as(data)
        return t


    def forward(
        self, data: Tensor, t: Optional[Tensor] = None, n_steps: Optional[int] = None, train_classifier: bool = False
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        """
        Compute an MC estimate of the continuous (when n_steps=None or 0) or discrete time KL loss.
        t is sampled randomly if None. If t is not None, expect t.shape == data.shape.
        """
        if isinstance(data, dict) and "cad_vec" in data and "conditions" in data:
            cad_vec = data['cad_vec']
            conditions = data['conditions'].to(torch.float32)

            t = self.sample_t(cad_vec, n_steps) if t is None else t
            input_params = self.bayesian_flow(cad_vec, t)   
            net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
            pred_sv: Tensor = self.classifier_net(net_inputs, t)
            
            mse_loss = self.mse_loss(conditions, pred_sv)
            
            if train_classifier:
                return mse_loss
            else:    
                output_params: Tensor = self.net(net_inputs, t, conditions)
                
                # compute KL loss in float32
                with torch.autocast(device_type=cad_vec.device.type if cad_vec.device.type != "mps" else "cpu", enabled=False):
                    if n_steps == 0 or n_steps is None:
                        loss = self.loss.cts_time_loss(cad_vec, output_params.float(), input_params, t)
                    else:
                        loss = self.loss.discrete_time_loss(cad_vec, output_params.float(), input_params, t, n_steps)
                    
            # loss shape is (batch_size, 1)            
            return loss.mean()      

    @torch.inference_mode()
    def compute_reconstruction_loss(self, data: Tensor) -> Tensor:
        t = torch.ones_like(data).float()
        input_params = self.bayesian_flow(data, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
        output_params: Tensor = self.net(net_inputs, t)
        # output_params = self.softmax(output_params)
        return self.loss.reconstruction_loss(data, output_params, input_params).flatten(start_dim=1).mean()

    @torch.inference_mode()
    def sample(self, data_shape: tuple, n_steps: int, conditions: Tensor, init_conditions: Tensor,
            guided_model: nn.Module = None) -> Tensor:
        
        device = next(self.parameters()).device
        conditions = conditions.to(torch.float32)
        
        # Mode flags
        CDE, UBI, GBF, num, H = True, True, True, 5, 5

        input_params = self.bayesian_flow.get_prior_input_params(data_shape, device)
        distribution_factory = self.loss.distribution_factory

        for i in range(1, n_steps + 1):
            t = torch.ones(*data_shape, device=device) * (i - 1) / n_steps
            net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
            output_params = self.net(net_inputs, t, conditions)
            alpha = self.bayesian_flow.get_alpha(i, n_steps)

            if CDE and UBI and GBF:
                def get_avg_sample():
                    sample_sum = None
                    for _ in range(H):
                        s = distribution_factory.get_dist(output_params, input_params, t).sample()
                        one_hot = F.one_hot(s.long(), 263)
                        sample_sum = one_hot if sample_sum is None else sample_sum + one_hot
                    return sample_sum / H

                input_params = self.average_theta_guided(
                    num=num,
                    get_output_sample=get_avg_sample,
                    update_func=self.bayesian_flow.update_input_params,
                    prob_func=self.bayesian_flow.get_sender_dist,
                    input_params=input_params,
                    t=t,
                    target=init_conditions,
                    guided_model=guided_model,
                    alpha=alpha,
                )

            elif not CDE and UBI and GBF:
                def get_sample():
                    return distribution_factory.get_dist(output_params, input_params, t).sample().reshape(*data_shape)

                input_params = self.average_theta_guided(
                    num=num,
                    get_output_sample=get_sample,
                    update_func=self.bayesian_flow.update_input_params,
                    prob_func=self.bayesian_flow.get_sender_dist_copy,
                    input_params=input_params,
                    t=t,
                    target=conditions,
                    guided_model=guided_model,
                    alpha=alpha,
                )

            elif not CDE and UBI and not GBF:
                def get_sample():
                    return distribution_factory.get_dist(output_params, input_params, t).sample().reshape(*data_shape)

                input_params = self.average_theta_simple(
                    num=num,
                    get_output_sample=get_sample,
                    update_func=self.bayesian_flow.update_input_params,
                    input_params=input_params,
                    alpha=alpha,
                )

            else:
                output_sample = distribution_factory.get_dist(output_params, input_params, t).sample().reshape(*data_shape)
                y = self.bayesian_flow.get_sender_dist_copy(output_sample, alpha).sample()
                input_params = self.bayesian_flow.update_input_params(input_params, y, alpha)

        # Final sampling
        t = torch.ones(*data_shape, device=device)
        output_params = self.net(self.bayesian_flow.params_to_net_inputs(input_params), t, conditions)
        output_sample = distribution_factory.get_dist(output_params, input_params, t).mode
        return output_sample.reshape(*data_shape)

    def compute_prob_from_guided_model(self, guided_model, theta, t, target):
        net_out = guided_model(theta, t)
        loc, scale = net_out[:, :2], torch.sqrt(net_out[:, 2:])
        dist = D.Normal(loc, scale, validate_args=False)
        logp = dist.log_prob(target)
        logp_mean = torch.mean(logp, dim=-1)
        prob = torch.exp(logp_mean).unsqueeze(1).unsqueeze(2)
        return prob

    def average_theta_guided(self, num, get_output_sample, update_func, prob_func, input_params, t, target, guided_model, alpha):
        theta_avg, probs = None, None
        for _ in range(num):
            output_sample = get_output_sample()
            y = prob_func(output_sample, alpha).sample()
            theta = update_func(input_params, y, alpha)[0]
            prob = self.compute_prob_from_guided_model(guided_model, theta, t, target)

            if theta_avg is None:
                theta_avg = theta * prob
                probs = prob
            else:
                theta_avg += theta * prob
                probs += prob

        theta_avg /= probs
        theta_avg = torch.clamp(theta_avg, max=100)
        theta_avg /= theta_avg.sum(-1, keepdims=True)
        return (theta_avg,)

    def average_theta_simple(self, num, get_output_sample, update_func, input_params, alpha):
        theta_sum = None
        for _ in range(num):
            output_sample = get_output_sample()
            y = self.bayesian_flow.get_sender_dist_copy(output_sample, alpha).sample()
            theta = update_func(input_params, y, alpha)[0]
            theta_sum = theta if theta_sum is None else theta_sum + theta

        theta_avg = theta_sum / num
        theta_avg = torch.clamp(theta_avg, max=100)
        theta_avg /= theta_avg.sum(-1, keepdims=True)
        return (theta_avg,)

