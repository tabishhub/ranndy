import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy as sp
from flax import linen as nn
from jax import random
from scipy.linalg import fractional_matrix_power

# Utility functions


def sortEig(A, evs=5, which="LM"):
    """
    Computes eigenvalues and eigenvectors of A and sorts them in decreasing lexicographic order.

    :param evs: number of eigenvalues/eigenvectors
    :return:    sorted eigenvalues and eigenvectors
    """
    n = A.shape[0]
    if evs < n:
        d, V = sp.sparse.linalg.eigs(A, evs, which=which)
    else:
        d, V = sp.linalg.eig(A)
    ind = d.argsort()[::-1]  # [::-1] reverses the list of indices
    return (d[ind], V[:, ind])


#
def qgedmd(X, psi, c0, c2, evs=5):
    """
    gEDMD for quantum

    params -
    X: data
    psi: basis functions
    c0: potential function

    returns -
    A: Hamiltonian approximation
    d, V: eigenvalues and eigenvectors of A
    """
    PsiX = psi(X)

    ddPsiX = psi.ddiff(X)  # Hessian (n,d,d,m)
    Vx = c0(X)  # potential (1, m)
    dPsiY = np.multiply(
        Vx, PsiX
    )  # potential multiplied basis (one of the terms in Hamiltonian operator)

    # for i in range(n):
    #     dPsiY[i, :] +=  c2(X) * np.trace(ddPsiX, axis1=1, axis2=2)

    dPsiY = dPsiY + c2(X) * np.trace(ddPsiX, axis1=1, axis2=2)

    C_0 = PsiX @ PsiX.T
    C_1 = PsiX @ dPsiY.T

    A = sp.linalg.pinv(C_0) @ C_1

    d, V = sortEig(A, evs, which="SR")

    return (A, d, V, C_0, C_1)


def inverse(x, epsilon=1e-10, ret_sqrt=False):
    """Utility function that returns the inverse of a matrix.

    Parameters-
    x: numpy array with shape [m,m]
        matrix to be inverted

    Returns-
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """

    # Calculate eigenvalues and eigenvectors
    eigval_all, eigvec_all = jnp.linalg.eigh(x)

    # Filter out eigenvalues below threshold and corresponding eigenvectors
    eigval = eigval_all[eigval_all > epsilon]
    eigvec = eigvec_all[:, eigval_all > epsilon]

    # Build the diagonal matrix with the filtered eigenvalues or square root
    # of the filtered eigenvalues according to the parameter
    if ret_sqrt:
        diag = jnp.diag(jnp.sqrt(1 / eigval))
    else:
        diag = jnp.diag(1 / eigval)

    # Rebuild the square root of the inverse matrix
    eigvec = eigvec.astype(jnp.float32)
    diag = diag.astype(jnp.float32)
    x_inv = jnp.matmul(eigvec, jnp.matmul(diag, eigvec.T))

    return x_inv


# ===============================================================================================================================
## Algorithms


class NeuralNetwork(nn.Module):
    hidden_sizes: list
    final_size: int
    activation: str = "tanh"
    init_weights_dist: str = "lecun_normal"
    init_weights_std: float = 1.0
    init_bias_dist: str = "zeros"
    init_bias_std: float = 1.0
    # randomized: bool = False
    batch_norm: bool = False
    vampnet: bool = False

    def activate(self, x):
        acts = {
            "relu": nn.relu,
            "sigmoid": nn.sigmoid,
            "tanh": nn.tanh,
            "elu": nn.elu,
            "softmax": lambda y: nn.softmax(y, axis=-1),
            "gelu": nn.gelu,
        }
        if self.activation not in acts:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return acts[self.activation](x)

    def get_kernel_init(self):
        inits = {
            "normal": nn.initializers.normal(self.init_weights_std),
            "uniform": nn.initializers.uniform(self.init_weights_std),
            "glorot_uniform": nn.initializers.glorot_uniform(),
            "glorot_normal": nn.initializers.glorot_normal(),
            "he_normal": nn.initializers.he_normal(),
            "he_uniform": nn.initializers.he_uniform(),
            "lecun_uniform": nn.initializers.lecun_uniform(),
            "lecun_normal": nn.initializers.lecun_normal(),
        }
        if self.init_weights_dist not in inits:
            raise ValueError(f"Unsupported weight init: {self.init_weights_dist}")
        return inits[self.init_weights_dist]

    def get_bias_init(self):
        inits = {
            "normal": nn.initializers.normal(self.init_bias_std),
            "uniform": nn.initializers.uniform(self.init_bias_std),
            "zeros": nn.initializers.zeros,
            "ones": nn.initializers.ones,
            "constant": nn.initializers.constant(self.init_bias_std),
        }
        if self.init_bias_dist not in inits:
            raise ValueError(f"Unsupported bias init: {self.init_bias_dist}")
        return inits[self.init_bias_dist]

    @nn.compact
    def __call__(self, x, training=True):
        # input: (features, batch)
        x = x.T  # (batch, features)

        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(
                hidden_size,
                kernel_init=self.get_kernel_init(),
                bias_init=self.get_bias_init(),
                name=f"dense_{i}",
            )(x)

            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=0.9,
                    epsilon=1e-5,
                    axis=-1,
                    name=f"bn_{i}",
                )(x)

            x = self.activate(x)

            # if self.randomized:
            #     x = jax.lax.stop_gradient(x)

        x = nn.Dense(
            self.final_size,
            kernel_init=self.get_kernel_init(),
            bias_init=self.get_bias_init(),
            name="final_layer",
        )(x)

        if self.vampnet:
            x = nn.softmax(x, axis=-1)
        else:
            x = self.activate(x)

        return x.T  # (features, batch)


class RaNNDy:
    def __init__(
        self,
        X,
        operator,
        hidden_sizes,
        final_size,
        activation="tanh",
        init_weights_dist: str = "normal",
        init_weights_std: float = 1.0,
        init_bias_dist: str = "normal",
        init_bias_std: float = 1.0,
        random_state=0,
        batch_norm=False,
    ):
        self.X = X
        self.operator = operator
        self.key = random.PRNGKey(random_state)

        # Create the neural network
        self.model = NeuralNetwork(
            hidden_sizes=hidden_sizes,
            final_size=final_size,
            activation=activation,
            init_weights_dist=init_weights_dist,
            init_weights_std=init_weights_std,
            init_bias_dist=init_bias_dist,
            init_bias_std=init_bias_std,
            batch_norm=batch_norm,
            vampnet=False,
        )

        # Initialize parameters
        self.params = self.model.init(self.key, jnp.ones(self.X.shape))

    def new_params(self, key):
        """Initializes new parameters for the model using a given random key."""
        return self.model.init(key, jnp.ones(self.X.shape))

    def compute_jacobian_nn(self, params, x):
        """Compute Jacobian for a single data point.

        Args:
            params: Model parameters
            x: Input of shape (d,)

        Returns:
            Jacobian of shape (final_size, d)
        """

        def model_apply_fn(x_):

            x_input = x_[..., None]  # (d, 1)
            y = self.model.apply(params, x_input, training=False)
            # y.shape: (final_size, 1)
            return jnp.squeeze(y, axis=1)  # (final_size,)

        return jax.jacobian(model_apply_fn, argnums=0)(x)

    def jacobian_all_nn(self, params, data):
        """Returns the Jacobian matrix evaluated for all the data points.

        Args:
            params: Model parameters
            data: Input of shape (d, m) where d=dimension, m=number of points

        Returns:
            Jacobian of shape (final_size, d, m)
        """
        return jax.vmap(
            self.compute_jacobian_nn,
            in_axes=(None, 1),  # Map over columns of data
            out_axes=2,  # Put batch dimension last
        )(params, data)

    def compute_hessian_nn(self, params, x):
        """Compute Hessian for a single data point.

        Args:
            params: Model parameters
            x: Input of shape (d,)

        Returns:
            Hessian of shape (final_size, d, d)
        """

        def model_apply_fn(x_):
            x_input = x_[..., None]  # (d, 1)
            y = self.model.apply(params, x_input, training=False)
            return jnp.squeeze(y, axis=1)  # (final_size,)

        return jax.hessian(model_apply_fn, argnums=0)(x)

    def hessian_all_nn(self, params, data):
        """Returns the Hessian matrix evaluated for all the data points.

        Args:
            params: Model parameters
            data: Input of shape (d, m) where d=dimension, m=number of points

        Returns:
            Hessian of shape (final_size, d, d, m)
        """
        return jax.vmap(
            self.compute_hessian_nn,
            in_axes=(None, 1),  # Map over columns of data
            out_axes=3,  # Put batch dimension last
        )(params, data)

    def koopman_eig_decomp(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        params=None,
        n: int = 5,
        epsilon: float = 1e-5,
    ):
        if params is None:
            params = self.params
        else:
            params = params

        # Random data matrices
        N_x = self.model.apply(params, X, training=False, mutable=False)
        N_y = self.model.apply(params, Y, training=False, mutable=False)

        ## subtract the mean
        # psi_x = psi_x - jnp.mean(psi_x, axis=1, keepdims=True)
        # psi_y = psi_y - jnp.mean(psi_y, axis=1, keepdims=True)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T  # (N_o, N_o)

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        # Solve the generalized eigenvalue problem

        A = jnp.linalg.pinv(C_xx_reg) @ C_xy  # Shape: (N_o, N_o)

        eigvals, eigvecs = jnp.linalg.eig(A)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take the top n eigenvectors
        E = eigvecs[:, :n].T  # shape (n, N_o)

        ## whitening transformation
        ECxE_T = E @ C_xx @ E.T  #

        # make E such that E C_xx E^T = I

        ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)  #

        # Apply whitening to E
        E_whitened = ECxE_T_inv_sqrt @ E

        return A, eigvals[:n], E_whitened

    def koopman_generator_eig_decomp(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray,
        params=None,
        n: int = 5,
        epsilon: float = 1e-5,
    ):
        if params is None:
            params = self.params
        else:
            params = params

        Y1 = Y  # kwargs.get("Y1")
        Z1 = Z  # kwargs.get("Z1")

        PsiX = self.model.apply(params, X, training=False, mutable=False)
        jacobX = self.jacobian_all_nn(params, X)

        dPsiY = jnp.einsum("ijk,jk->ik", jacobX, Y1)

        N_o = PsiX.shape[0]  # number of basis functions
        hessX = self.hessian_all_nn(params, X)  # second-order derivatives
        S = jnp.einsum("ijk,ljk->ilk", Z1, Z1)  # sigma \cdot sigma^T
        # dPsiY = self.compute_dPsiY(dPsiY, hessX, S, n)
        for i in range(N_o):
            dPsiY = dPsiY.at[i, :].add(
                0.5 * jnp.sum(hessX[i, :, :, :] * S, axis=(0, 1))
            )

        N_x_dot = dPsiY
        N_x = PsiX

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_x_dot.T  # (N_o, N_o)
        # Regularization
        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])

        # Compute the linear operator A = C_yx @ pinv(C_xx)
        A = jnp.linalg.pinv(C_xx_reg) @ C_xy  # Shape: (N_o, N_o)
        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = jnp.linalg.eig(A)

        # Sort eigenvalues in ascending order
        idx = np.argsort(eigvals)[::1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take the top n eigenvectors
        E = eigvecs[:, :n].T  # shape (n, N_o)

        ## whitening transformation
        ECxE_T = E @ C_xx @ E.T  #

        # make E such that E C_xx E^T = I

        ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)  #

        # Apply whitening to E
        E_whitened = ECxE_T_inv_sqrt @ E

        return A, eigvals[:n], E_whitened

    def schrodinger_eig_decomp(
        self,
        X: jnp.ndarray,
        params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        if params is None:
            params = self.params
        else:
            params = params

        c0 = kwargs.get("c0")
        c2 = kwargs.get("c2")
        PsiX = self.model.apply(params, X, training=False, mutable=False)
        Hessian = self.hessian_all_nn(params, X)
        Vx = c0(X)
        dPsiY = (
            Vx * PsiX
        )  # potential multiplied basis (one of the terms in Hamiltonian operator)

        # use jnp.trace for the Hessian trace
        lap = jnp.trace(Hessian, axis1=1, axis2=2)  # shape (m,)

        dPsiY = dPsiY + c2(X) * lap
        N_x_dot = dPsiY
        N_x = PsiX

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_x_dot.T  # (N_o, N_o)
        # Regularization
        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        # Compute the linear operator A = C_yx @ pinv(C_xx)
        A = jnp.linalg.pinv(C_xx_reg) @ C_xy  # Shape: (N_o, N_o)
        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = jnp.linalg.eig(A)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take the top n eigenvectors
        E = eigvecs[:, :n].T  # shape (n, N_o)

        ## whitening transformation
        ECxE_T = E @ C_xx @ E.T  #

        # make E such that E C_xx E^T = I

        ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)  #

        # Apply whitening to E
        E_whitened = ECxE_T_inv_sqrt @ E

        return A, eigvals[:n], E_whitened

    def forward_backward_eig_decomp(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        params=None,
        n: int = 5,
        epsilon: float = 1e-5,
    ):
        if params is None:
            params = self.params
        else:
            params = params

        # Random data matrices
        N_x = self.model.apply(params, X, training=False, mutable=False)
        N_y = self.model.apply(params, Y, training=False, mutable=False)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T  # (N_o, N_o)
        C_yy = N_y @ N_y.T

        # Regularization
        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        # Compute the linear operator A = C_yx @ pinv(C_xx)
        A = (
            jnp.linalg.pinv(C_xx_reg) @ C_xy @ jnp.linalg.pinv(C_yy_reg) @ C_xy.T
        )  # Shape: (N_o, N_o)
        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = jnp.linalg.eig(A)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take the top n eigenvectors
        E = eigvecs[:, :n].T  # shape (n, N_o)
        ## whitening transformation
        ECxE_T = E @ C_xx @ E.T  #

        # make E such that E C_xx E^T = I

        ECxE_T_inv_sqrt = fractional_matrix_power(ECxE_T, -0.5)  #

        # Apply whitening to E
        E_whitened = ECxE_T_inv_sqrt @ E

        return A, eigvals[:n], E_whitened

    def operator_eig_decomp(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray = None,
        Z: jnp.ndarray = None,
        params=None,
        n: int = 5,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        """
        Calculates the eigenvalues and eigenvectors of the approximated operator matrix.
        Params:
        n - number of eigenvalues and eigenvectors to return
        epsilon - regularization parameter
        kwargs - arguments for different operators

        Returns:
        eigvals - eigenvalues of the operator matrix
        eigvecs - eigenvectors of the operator matrix
        """
        if params is None:
            params = self.params
        else:
            params = params
        # for Koopman operator
        if self.operator == "koopman":
            A, eigvals, E = self.koopman_eig_decomp(
                X=X, Y=Y, params=params, n=n, epsilon=epsilon
            )
            return A, eigvals, E

        # for Koopman generator
        elif self.operator == "koopman_generator":
            A, eigvals, E = self.koopman_generator_eig_decomp(
                X=X, Y=Y, Z=Z, params=params, n=n, epsilon=epsilon
            )
            return A, eigvals, E

        # for Schrodinger operator
        elif self.operator == "schrodinger":
            A, eigvals, E = self.schrodinger_eig_decomp(
                X=X, params=params, n=n, epsilon=epsilon, **kwargs
            )
            return A, eigvals, E

        # for forward-backward operator
        elif self.operator == "forward_backward":
            A, eigvals, E = self.forward_backward_eig_decomp(
                X=X, Y=Y, params=params, n=n, epsilon=epsilon
            )
            return A, eigvals, E

        else:
            raise ValueError(
                f"Operator {self.operator} is not available. Please select one of {'koopman', 'koopman_generator', 'schrodinger', 'forward_backward'}."
            )

    def ranks_cov_matrices(self, X, Y, epsilon, params=None):
        if params is None:
            params = self.params
        else:
            params = params

        # Random data matrices
        N_x = self.model.apply(params, X, training=False, mutable=False)
        N_y = self.model.apply(params, Y, training=False, mutable=False)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T  # (N_o, N_o)
        C_yy = N_y @ N_y.T

        # Regularization
        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        print(
            "Rank of C_00 and C_00_reg matrix: ",
            jnp.linalg.matrix_rank(C_xx),
            jnp.linalg.matrix_rank(C_xx_reg),
        )
        print("Rank of C_01 matrix: ", jnp.linalg.matrix_rank(C_xy))
        print(
            "Rank of C_11 and C_11_reg matrix: ",
            jnp.linalg.matrix_rank(C_yy),
            jnp.linalg.matrix_rank(C_yy_reg),
        )
        return None

    def ensemble_models(
        self,
        n_models,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Z: jnp.ndarray = None,
        domain: jnp.ndarray = None,
        n: int = 5,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        """
        Computes the ensemble of models using the input data.
        Params:
        n_models - number of models in the ensemble
        X - Input data
        Y - Input data
        Z - input data for the Koopman generator (optional)
        n - number of eigenvalues and eigenvectors to return
        epsilon - regularization parameter
        kwargs - arguments for different operators

        Returns:
        H_aug - Ensemble of models
        """
        if domain is None:
            domain = X

        ensem_eigvals = np.zeros((n, n_models))
        ensem_eigfuncs = np.zeros((n, domain.shape[1], n_models))
        # keys = random.split(self.key, n_models)
        for i in range(n_models):
            params_n = self.new_params(random.PRNGKey(i))
            A, eigvals, eigvecs = self.operator_eig_decomp(
                X, Y, None, params=params_n, n=n, epsilon=epsilon, **kwargs
            )
            ensem_eigvals[:, i] = eigvals
            eigfuncs = self.eigenfunctions(eigvecs, domain, params=params_n)
            ensem_eigfuncs[:, :, i] = eigfuncs

        V_ref = ensem_eigfuncs[:, :, 0]
        for j in range(n):
            for i in range(1, n_models):
                V1 = ensem_eigfuncs[j, :, i]
                V2 = -ensem_eigfuncs[j, :, i]
                e1 = np.linalg.norm(V_ref[j, :] - V1)
                e2 = np.linalg.norm(V_ref[j, :] - V2)
                if e2 < e1:
                    ensem_eigfuncs[j, :, i] = -ensem_eigfuncs[j, :, i]

        avg_eigvals = np.mean(ensem_eigvals, axis=1)
        avg_eigfuncs = np.mean(ensem_eigfuncs, axis=2)
        eigvals_var = ensem_eigvals.std(axis=1)
        eigfuncs_var = ensem_eigfuncs.std(axis=2)
        eigvals_plus = avg_eigvals + np.abs(eigvals_var)
        eigvals_minus = avg_eigvals - np.abs(eigvals_var)
        eigfuncs_plus = avg_eigfuncs + np.abs(eigfuncs_var)
        eigfuncs_minus = avg_eigfuncs - np.abs(eigfuncs_var)

        return (
            avg_eigvals,
            avg_eigfuncs,
            eigvals_plus,
            eigvals_minus,
            eigfuncs_plus,
            eigfuncs_minus,
        )

    def eigenfunctions(self, eigvecs, domain: jnp.ndarray = None, params=None):
        """
        Computes the eigenfunctions of the operator using the eigenvectors and input data.
        Params:
        eigvecs - Eigenvectors of the operator
        domain - Domain for the eigenfunctions (optional)

        Returns:
        eigfuncs - Eigenfunctions of the operator
        """
        if params is None:
            params = self.params

        else:
            params = params

        if domain is None:
            H_aug = self.model.apply(params, self.X, training=False, mutable=False)
        else:
            H_aug = self.model.apply(params, domain, training=False, mutable=False)
        eigfuncs = eigvecs @ H_aug

        return eigfuncs


# ====================================================================================================================
## VAMPNets in JAX


class VAMPNets:
    def __init__(
        self,
        X,  # Input data for the neural network
        hidden_sizes,
        final_size,  # output layer size
        activation="tanh",  # activation function
        init_weights_dist: str = "lecun_normal",
        init_weights_std: float = 1.0,  # Standard deviation for weight initialization
        init_bias_dist: str = "zeros",
        init_bias_std: float = 1.0,  # Standard deviation for bias initialization
        # randomized: bool = False,  # Whether to use randomized weights initialization
        batch_norm: bool = False,  # Whether to use batch normalization
        random_state=0,  # Random state for reproducibility
    ):
        self.X = X
        self.key = random.PRNGKey(random_state)
        self.model = NeuralNetwork(
            hidden_sizes,
            final_size,
            activation,
            init_weights_dist,
            init_weights_std,
            init_bias_dist,
            init_bias_std,
            # randomized,
            batch_norm,
            vampnet=True,
        )
        self.params = self.model.init(self.key, jnp.ones(self.X.shape), training=True)
        # self.params = self.variables["params"]
        # self.batch_stats = self.variables.get("batch_stats", {})

    def optimizer(self, optim="adam", lr=1e-3):  # , freeze_layers=None):
        if optim == "adam":
            base = optax.adam(lr)
        elif optim == "gd":
            base = optax.sgd(lr)
        elif optim == "nesterov":
            base = optax.sgd(lr, momentum=0.1, nesterov=True)
        elif optim == "rmsprop":
            base = optax.rmsprop(lr)
        else:
            raise ValueError(f"Unknown optimizer {optim}")

        return base

    def cost_vamp2(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        epsilon: float = 1e-5,
    ):
        """
        Calculates the VAMP-2 score with respect to the new basis functions.

        Parameters:
        params : jnp.ndarray
            Parameters of the model.
        X : jnp.ndarray
            Input data
        Y : jnp.ndarray
            Input data
        Returns:
        VAMP-2 score
        """

        psi_x = self.model.apply(
            params, X
        )  # , training=False, mutable=["batch_stats"])[0]
        psi_y = self.model.apply(
            params, Y
        )  # , training=False, mutable=["batch_stats"])[0]

        # psi_x = psi_x - jnp.mean(psi_x, axis=1, keepdims=True)
        # psi_y = psi_y - jnp.mean(psi_y, axis=1, keepdims=True)

        data_points = psi_x.shape[1]

        # Calculate the covariance matrices
        cov_01 = 1 / (data_points - 1) * jnp.matmul(psi_x, psi_y.T)
        cov_00 = 1 / (data_points - 1) * jnp.matmul(psi_x, psi_x.T) + epsilon * jnp.eye(
            psi_x.shape[0]
        )
        cov_11 = 1 / (data_points - 1) * jnp.matmul(psi_y, psi_y.T) + epsilon * jnp.eye(
            psi_x.shape[0]
        )

        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = inverse(cov_00, ret_sqrt=True)
        cov_11_inv = inverse(cov_11, ret_sqrt=True)

        vamp_matrix = jnp.matmul(jnp.matmul(cov_00_inv, cov_01), cov_11_inv)

        # For sum of eigenvalues
        # vamp_matrix = jnp.matmul(cov_00_inv, cov_01)
        # vamp_score = jnp.linalg.eigh(vamp_matrix)[0]  # sortEig(np.array(vamp_matrix), evs=3)[0]

        vamp_score = jnp.linalg.norm(vamp_matrix, "fro")

        return -jnp.square(vamp_score)  # -jnp.sum(jnp.real(vamp_score[:5]))

    def training(self, X, Y, n=5, epochs=100, optim="adam", lr=1e-3, epsilon=1e-5):
        tx = self.optimizer(optim=optim, lr=lr)
        opt_state = tx.init(self.params)

        def train_step(params, opt_state, X, Y):
            grads = jax.grad(self.cost_vamp2)(params, X, Y, epsilon)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            loss = self.cost_vamp2(params=params, X=X, Y=Y, epsilon=epsilon)
            return params, opt_state, loss

        params = self.params
        losses = []
        for epoch in range(epochs):
            params, opt_state, _ = train_step(params, opt_state, X, Y)
            _, eigvals, _ = self.koopman_approximation(params=params, X=X, Y=Y, n=n)
            losses.append(jnp.sum(jnp.real(eigvals)))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {-jnp.sum(jnp.real(eigvals)):.4f}")

        self.params = params
        return params, losses

    def koopman_approximation(
        self,
        params,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        n: int = 5,
        operator: str = "koopman",
        epsilon: float = 1e-5,
    ):
        """
        Calculates the Koopman operator approximation.
        Params:
        n - number of eigenvalues and eigenvectors to return
        epsilon - regularization parameter

        Returns:
        eigvals - eigenvalues of the operator matrix
        eigvecs - eigenvectors of the operator matrix
        """

        # Random data matrices
        N_x = self.model.apply(params, X)  # , training=False, mutable=False)
        N_y = self.model.apply(params, Y)  # , training=False, mutable=False)

        ## subtract the mean
        # N_x = N_x - jnp.mean(N_x, axis=1, keepdims=True)
        # N_y = N_y - jnp.mean(N_y, axis=1, keepdims=True)

        C_xx = N_x @ N_x.T  # (N_o, N_o)
        C_xy = N_x @ N_y.T
        C_yy = N_y @ N_y.T

        C_xx_reg = C_xx + epsilon * jnp.identity(C_xx.shape[0])
        C_yy_reg = C_yy + epsilon * jnp.identity(C_yy.shape[0])

        if operator == "koopman":
            # Compute the linear operator A = C_yx @ pinv(C_xx)
            A = jnp.linalg.pinv(C_xx_reg) @ C_xy  # Shape: (N_o, N_o)

        elif operator == "forward_backward":
            A = jnp.linalg.pinv(C_xx_reg) @ C_xy @ jnp.linalg.pinv(C_yy_reg) @ C_xy.T

        eigvals, eigvecs = jnp.linalg.eig(A)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return A, eigvals[:n], eigvecs[:, :n]

    def eigenfunctions(
        self, params: jnp.ndarray, eigvecs: jnp.ndarray, domain: jnp.ndarray = None
    ):
        """
        Computes the eigenfunctions of the operator using the eigenvectors and input data.
        Params:
        eigvecs - Eigenvectors of the operator
        domain - Domain for the eigenfunctions
        params - Parameters of the model

        Returns:
        eigfuncs - Eigenfunctions of the operator
        """

        if domain is None:
            H_aug = self.model.apply(params, self.X)  # , training=False, mutable=False)
        else:
            H_aug = self.model.apply(params, domain)  # , training=False, mutable=False)
        eigfuncs = eigvecs @ H_aug

        return eigfuncs
